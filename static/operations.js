const node = id => document.getElementById(id);
const formatScore = value => value ? `${value.passed} / ${value.total}` : '—';
async function refresh() {
    try {
        const response = await fetch('/api/metrics', {cache:'no-store'});
        if (!response.ok) throw new Error();
        const m = await response.json();
        for (const key of ['requests','errors','insufficient']) node(key).textContent = m[key];
        node('latency').textContent = m.p95_ms === null ? '—' : (m.p95_ms < 1000 ? Math.round(m.p95_ms)+' ms' : (m.p95_ms/1000).toFixed(1)+' s');
        node('scores').replaceChildren();
        for (const [key,label] of Object.entries({answer_correctness:'Answer correctness',citation_accuracy:'Citation accuracy',abstention_recall:'Insufficient-evidence recall',evidence_decisions:'Evidence decision accuracy'})) {
            const row = node('scores').insertRow();
            for (const text of [label, ...['baseline','guarded'].map(mode => formatScore(m.evals[mode]?.scores[key]))]) row.insertCell().textContent = text;
        }
        const run = m.evals.guarded || m.evals.baseline;
        node('eval-info').textContent = run ? run.model+' · '+new Date(run.finished_at).toLocaleString() : 'No completed evaluation yet.';
        node('failures').replaceChildren();
        node('empty').textContent = m.failures.length ? 'Select a failure to inspect its trace.' : 'No errors in the last 24 hours.';
        for (const run of m.failures) {
            const a = document.createElement('a');
            a.className = 'failure';
            a.href = '/traces/'+encodeURIComponent(run.trace_id);
            a.textContent = `${run.status} · /${run.operation} · ${run.reason} · ${new Date(run.started*1000).toLocaleTimeString()}`;
            node('failures').append(a);
        }
        node('fresh').textContent = 'Updated '+new Date().toLocaleTimeString();
    } catch {
        node('fresh').textContent = 'Refresh failed. Reconnecting…';
    } finally {
        setTimeout(refresh, 5000);
    }
}
refresh();
