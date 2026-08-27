import re
import tempfile
import unittest

from app import bcrypt, create_app, db
from app.models import JobDescription, ScreeningCriteria, User
from config import resolve_secret_key


class SecurityTestConfig:
    TESTING = True
    SECRET_KEY = 'security-test-key-with-at-least-32-characters'
    SQLALCHEMY_DATABASE_URI = 'sqlite://'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = tempfile.gettempdir()
    WTF_CSRF_ENABLED = False
    RATELIMIT_ENABLED = False


class CsrfTestConfig(SecurityTestConfig):
    WTF_CSRF_ENABLED = True


class RateLimitTestConfig(SecurityTestConfig):
    RATELIMIT_ENABLED = True
    RATELIMIT_STORAGE_URI = 'memory://'
    RATELIMIT_HEADERS_ENABLED = True


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(SecurityTestConfig)
        self.context = self.app.app_context()
        self.context.push()
        db.drop_all()
        db.create_all()

        self.users = {}
        for role in ('hr', 'manager', 'admin'):
            user = User(username=role, password_hash='unused', role=role)
            db.session.add(user)
            db.session.flush()
            self.users[role] = user.id

        job = JobDescription(
            title='Security Test Job',
            required_skills='Python',
            created_by=self.users['manager'],
        )
        db.session.add(job)
        db.session.flush()
        db.session.add(ScreeningCriteria(job_id=job.id))
        db.session.commit()
        self.job_id = job.id
        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.context.pop()

    def login_as(self, role):
        with self.client.session_transaction() as session:
            session['_user_id'] = str(self.users[role])
            session['_fresh'] = True

    def test_hr_cannot_manage_or_delete_jobs(self):
        self.login_as('hr')
        self.assertEqual(self.client.get('/jobs/create').status_code, 403)
        self.assertEqual(
            self.client.post(f'/jobs/{self.job_id}/delete').status_code,
            403,
        )
        self.assertIsNotNone(db.session.get(JobDescription, self.job_id))

    def test_manager_can_manage_but_cannot_delete_jobs(self):
        self.login_as('manager')
        self.assertEqual(self.client.get('/jobs/create').status_code, 200)
        self.assertEqual(
            self.client.post(f'/jobs/{self.job_id}/delete').status_code,
            403,
        )

    def test_admin_can_delete_jobs(self):
        self.login_as('admin')
        response = self.client.post(f'/jobs/{self.job_id}/delete')
        self.assertEqual(response.status_code, 302)
        self.assertIsNone(db.session.get(JobDescription, self.job_id))

    def test_only_admin_can_bulk_delete_candidates(self):
        self.login_as('hr')
        self.assertEqual(
            self.client.post('/screening_results/delete_all').status_code,
            403,
        )

    def test_role_command_assigns_an_approved_role(self):
        runner = self.app.test_cli_runner()
        result = runner.invoke(args=['set-user-role', 'manager', 'admin'])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(db.session.get(User, self.users['manager']).role, 'admin')

    def test_bootstrap_admin_command_creates_an_idempotent_production_admin(self):
        runner = self.app.test_cli_runner()
        command_env = {
            'INITIAL_ADMIN_USERNAME': 'deployed-admin',
            'INITIAL_ADMIN_PASSWORD': 'strong-deployment-password',
        }

        first = runner.invoke(args=['bootstrap-admin'], env=command_env)
        second = runner.invoke(args=['bootstrap-admin'], env=command_env)

        self.assertEqual(first.exit_code, 0, first.output)
        self.assertEqual(second.exit_code, 0, second.output)
        user = User.query.filter_by(username='deployed-admin').one()
        self.assertEqual(user.role, 'admin')
        self.assertTrue(bcrypt.check_password_hash(
            user.password_hash,
            command_env['INITIAL_ADMIN_PASSWORD'],
        ))
        self.assertEqual(User.query.filter_by(username='deployed-admin').count(), 1)

    def test_bootstrap_admin_command_rejects_a_weak_password(self):
        result = self.app.test_cli_runner().invoke(
            args=['bootstrap-admin'],
            env={
                'INITIAL_ADMIN_USERNAME': 'deployed-admin',
                'INITIAL_ADMIN_PASSWORD': 'too-short',
            },
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn('at least 12 characters', result.output)
        self.assertIsNone(User.query.filter_by(username='deployed-admin').first())

    def test_production_requires_a_strong_secret(self):
        with self.assertRaises(RuntimeError):
            resolve_secret_key('production', '')
        with self.assertRaises(RuntimeError):
            resolve_secret_key('production', 'too-short')
        with self.assertRaises(RuntimeError):
            resolve_secret_key(
                'production',
                'replace-with-a-random-secret-of-at-least-32-characters',
            )

    def test_development_secret_is_random_when_not_configured(self):
        first = resolve_secret_key('development', '')
        second = resolve_secret_key('development', '')
        self.assertGreaterEqual(len(first), 32)
        self.assertNotEqual(first, second)


class CsrfTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(CsrfTestConfig)
        self.context = self.app.app_context()
        self.context.push()
        db.drop_all()
        db.create_all()
        db.session.add(User(
            username='csrf-user',
            password_hash=bcrypt.generate_password_hash('correct-password').decode('utf-8'),
            role='hr',
        ))
        db.session.commit()
        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.context.pop()

    def test_login_rejects_post_without_csrf_token(self):
        response = self.client.post(
            '/login',
            data={'username': 'someone', 'password': 'wrong'},
        )
        self.assertEqual(response.status_code, 400)

    def test_login_form_contains_csrf_token(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertRegex(
            response.get_data(as_text=True),
            re.compile(r'name="csrf_token"\s+value="[^"]+"'),
        )

    def test_valid_csrf_token_allows_login(self):
        page = self.client.get('/login').get_data(as_text=True)
        token = re.search(r'name="csrf_token"\s+value="([^"]+)"', page).group(1)
        response = self.client.post(
            '/login',
            data={
                'csrf_token': token,
                'username': 'csrf-user',
                'password': 'correct-password',
            },
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers['Location'], '/dashboard')


class LoginRateLimitTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(RateLimitTestConfig)
        self.context = self.app.app_context()
        self.context.push()
        db.drop_all()
        db.create_all()
        self.client = self.app.test_client()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.context.pop()

    def test_sixth_login_attempt_is_rate_limited(self):
        for attempt in range(6):
            response = self.client.post(
                '/login',
                data={'username': 'unknown', 'password': 'wrong'},
                environ_overrides={'REMOTE_ADDR': '198.51.100.42'},
            )
            if attempt < 5:
                self.assertEqual(response.status_code, 200)
        self.assertEqual(response.status_code, 429)
        self.assertIn('Retry-After', response.headers)


if __name__ == '__main__':
    unittest.main()
