"""Executive Summary routes.

Aggregates uploaded resumes and screening results per job, grouped by month,
so HR can see which positions attracted the most applicants over time.

No new tables are required: Resume.job_id / Resume.upload_date and
ScreeningResult.job_id / ScreeningResult.screened_at already carry
everything needed for these aggregates.
"""

from collections import OrderedDict, defaultdict
from datetime import datetime, timezone

from flask import Blueprint, current_app, render_template, request
from flask_login import login_required

from app.models import JobDescription, Resume, ScreeningResult
from app.utils.device import current_device_id

try:  # pragma: no cover - fallback when tzdata is unavailable
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    ZoneInfo = None
    ZoneInfoNotFoundError = Exception

summary_bp = Blueprint('summary', __name__)


def _display_tz():
    """Resolve the app's display timezone, mirroring app/__init__.py."""
    tz_name = current_app.config.get('DISPLAY_TIMEZONE', 'Asia/Manila')
    if ZoneInfo is None:
        return timezone.utc
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        return timezone.utc


def _to_local(value, tz):
    """Treat naive DB timestamps as UTC, then convert to display timezone."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(tz)


def _month_key(dt):
    """Sortable YYYY-MM key."""
    return f"{dt.year:04d}-{dt.month:02d}"


def _month_label(month_key):
    """Human label, e.g. '2026-08' -> 'August 2026'."""
    year, month = month_key.split('-')
    return datetime(int(year), int(month), 1).strftime('%B %Y')


def build_summary(months_limit=12, device_id=None):
    """Build the executive summary payload.

    Returns a dict with:
      - months:        [{key, label}] newest first
      - by_month:      {month_key: [job rows sorted most -> least applicants]}
      - totals:        overall counters
      - trend:         [{label, total}] oldest -> newest, for the chart
    """
    tz = _display_tz()
    owner = device_id or current_device_id()

    jobs = {job.id: job for job in JobDescription.query.all()}

    # ---- resumes uploaded, bucketed by (month, job) ----
    uploads = defaultdict(lambda: defaultdict(int))
    for resume in Resume.query.filter_by(device_id=owner).all():
        local_dt = _to_local(resume.upload_date, tz)
        if local_dt is None:
            continue
        uploads[_month_key(local_dt)][resume.job_id] += 1

    # ---- screening results, bucketed by (month, job) ----
    screened = defaultdict(lambda: defaultdict(int))
    qualified = defaultdict(lambda: defaultdict(int))
    score_sum = defaultdict(lambda: defaultdict(float))

    for result in ScreeningResult.query.filter_by(device_id=owner).all():
        local_dt = _to_local(result.screened_at, tz)
        if local_dt is None:
            continue
        key = _month_key(local_dt)
        screened[key][result.job_id] += 1
        score_sum[key][result.job_id] += (result.fit_score or 0.0)
        if result.recommendation_label == 'Qualified':
            qualified[key][result.job_id] += 1

    # ---- assemble per-month rows ----
    all_months = sorted(set(uploads) | set(screened), reverse=True)
    all_months = all_months[:months_limit]

    by_month = OrderedDict()
    for month in all_months:
        job_ids = set(uploads[month]) | set(screened[month])
        rows = []
        for job_id in job_ids:
            job = jobs.get(job_id)
            applicants = uploads[month].get(job_id, 0)
            screened_count = screened[month].get(job_id, 0)
            qualified_count = qualified[month].get(job_id, 0)
            avg_fit = (
                round(score_sum[month][job_id] / screened_count, 1)
                if screened_count else None
            )
            rows.append({
                'job_id': job_id,
                'job_title': job.title if job else 'Deleted job',
                'job_exists': job is not None,
                'applicants': applicants,
                'screened': screened_count,
                'qualified': qualified_count,
                'qualified_rate': (
                    round(qualified_count / screened_count * 100, 1)
                    if screened_count else None
                ),
                'avg_fit_score': avg_fit,
            })

        # most applied -> least applied
        rows.sort(key=lambda r: (-r['applicants'], -r['screened'], r['job_title'].lower()))

        month_total = sum(r['applicants'] for r in rows)
        for row in rows:
            row['share'] = (
                round(row['applicants'] / month_total * 100, 1) if month_total else 0.0
            )

        by_month[month] = {
            'label': _month_label(month),
            'rows': rows,
            'total_applicants': month_total,
            'total_screened': sum(r['screened'] for r in rows),
            'total_qualified': sum(r['qualified'] for r in rows),
            'top_job': rows[0] if rows else None,
        }

    months = [{'key': m, 'label': by_month[m]['label']} for m in all_months]

    # oldest -> newest for the trend chart
    trend = [
        {'label': by_month[m]['label'], 'total': by_month[m]['total_applicants']}
        for m in reversed(all_months)
    ]

    totals = {
        'total_applicants': sum(v['total_applicants'] for v in by_month.values()),
        'total_screened': sum(v['total_screened'] for v in by_month.values()),
        'total_qualified': sum(v['total_qualified'] for v in by_month.values()),
        'months_covered': len(all_months),
    }

    return {
        'months': months,
        'by_month': by_month,
        'totals': totals,
        'trend': trend,
    }


@summary_bp.route('/summary')
@login_required
def executive_summary():
    """Executive Summary: applicants per job, per month (most -> least)."""
    data = build_summary()

    selected = request.args.get('month')
    month_keys = [m['key'] for m in data['months']]
    if selected not in month_keys:
        selected = month_keys[0] if month_keys else None

    current = data['by_month'].get(selected) if selected else None

    return render_template(
        'summary/executive_summary.html',
        months=data['months'],
        selected_month=selected,
        current=current,
        totals=data['totals'],
        trend=data['trend'],
    )
