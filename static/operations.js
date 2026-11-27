const node = id => document.getElementById(id);
const formatScore = value => value ? `${value.passed} / ${value.total}` : '—';
const percent = value => value == null ? '—' : (100 * value).toFixed(1) + '%';
const latency = value => value == null ? '—' : value < 1000 ? value.toFixed(1)+' ms' : (value/1000).toFixed(2)+' s';
const usd = value => value == null ? '—' : '$'+value.toFixed(8);
const number = value => value == null ? '—' : value.toLocaleString();
function qualityValue(key, value) {
    if (value == null) return '—';
    if (key === 'retrieved_count') return value.toFixed(1);
    if (key.startsWith('retrieval_cosine')) return value.toFixed(3);
    return percent(value);
}
let trafficChart;
let qualityChart;
async function refresh() {
    try {
        const response = await fetch('/api/metrics', {cache:'no-store'});
        if (!response.ok) throw new Error();
        const m = await response.json();
        for (const key of ['requests','errors','insufficient']) node(key).textContent = m[key];
        node('latency').textContent = m.p95_ms === null ? '—' : (m.p95_ms < 1000 ? Math.round(m.p95_ms)+' ms' : (m.p95_ms/1000).toFixed(1)+' s');
        for (const key of ['total_requests','requests_per_minute','success_rate','error_rate','mean_ms','p50_ms','p99_ms','guardrail_trigger_rate','abstention_rate']) {
            node('op-'+key).textContent = key.endsWith('_rate') ? percent(m[key]) : key.endsWith('_ms') ? latency(m[key]) : number(m[key]);
        }
        node('op-ttft').textContent = latency(m.provider_ttft.mean_ms)+' / '+latency(m.provider_ttft.p95_ms)+' · n='+m.provider_ttft.n;
        node('provider-usage').replaceChildren();
        for (const phase of ['request','evaluation']) {
            const u = m.usage[phase];
            const row = node('provider-usage').insertRow();
            for (const value of [phase === 'request' ? 'Answer + guardrails' : 'Background Ragas',
                ...['input_tokens','output_tokens','thinking_tokens','cached_tokens','total_tokens'].map(k=>number(u[k])),
                u.measured_calls+' / '+u.calls, usd(u.estimated_cost_usd)+' · '+u.priced_calls+' priced calls']) row.insertCell().textContent = value;
        }
        node('cost-per-query').textContent = 'Mean estimated cost per settled, fully priced query, including background evaluation · '+usd(m.usage.cost_per_query.mean_usd)+' · n='+m.usage.cost_per_query.n;
        node('pricing').textContent = m.pricing.model+' · '+m.pricing.basis+' Recorded tokens may be incomplete for failed calls.';
        node('stage-latency').replaceChildren();
        for (const [name,s] of Object.entries(m.stages)) {
            const row = node('stage-latency').insertRow();
            for (const value of [name.replaceAll('_',' '),s.n,latency(s.mean_ms),latency(s.p95_ms)]) row.insertCell().textContent=value;
        }
        node('rag-window').textContent = m.rag_queries + ' query requests in the last 24 hours. Each mean includes only queries with a measurement.';
        node('eval-jobs').textContent = Object.entries(m.eval_jobs).map(([s,n])=>s.replaceAll('_',' ')+' '+n).join(' · ') || 'Ragas scores appear after delivered answers are evaluated.';
        node('guardrails').replaceChildren();
        for (const [reason, count] of Object.entries(m.guardrails)) {
            const p = document.createElement('p');
            p.className = 'sub';
            p.textContent = reason.replaceAll('_',' ') + ' · ' + count;
            node('guardrails').append(p);
        }
        if (!Object.keys(m.guardrails).length) node('guardrails').textContent = 'No insufficient-evidence decisions in this window.';
        for (const [key, score] of Object.entries(m.rag_live)) {
            node('mean-'+key).textContent = qualityValue(key,score.mean);
            node('count-'+key).textContent = score.n;
        }
        const queryList = node('queries');
        queryList.replaceChildren();
        for (const q of m.recent_queries) {
            const row = queryList.insertRow();
            const link = document.createElement('a');
            link.href = '/traces/'+encodeURIComponent(q.trace_id);
            link.textContent = new Date(q.started*1000).toLocaleTimeString();
            row.insertCell().append(link);
            for (const text of [q.outcome.replaceAll('_',' '), percent(q.quality.ragas_answer_relevancy), percent(q.quality.ragas_faithfulness), (q.eval_status || 'not scheduled').replaceAll('_',' ')]) row.insertCell().textContent = text;
        }
        if (!m.recent_queries.length) {
            const cell = queryList.insertRow().insertCell();
            cell.colSpan = 5;
            cell.textContent = 'No queries in this window.';
        }
        node('scores').replaceChildren();
        for (const [key,label] of Object.entries({answer_correctness:'Answer correctness (gold fact coverage)',citation_accuracy:'Citation accuracy (page + evidence)',abstention_recall:'Insufficient-evidence recall',abstention_accuracy:'Abstention accuracy'})) {
            const row = node('scores').insertRow();
            for (const text of [label, ...['baseline','guarded'].map(mode => formatScore(m.evals[mode]?.scores[key]))]) row.insertCell().textContent = text;
        }
        const run = m.evals.guarded || m.evals.baseline;
        for (const mode of ['baseline','guarded']) {
            for (const [name,s] of Object.entries(m.evals[mode]?.rag_scores || {})) {
                const cell = node('eval-'+mode+'-'+name);
                if (cell && s.n) cell.textContent = (name === 'reciprocal_rank' ? s.mean.toFixed(3) : percent(s.mean)) + ' · n='+s.n;
            }
        }
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
        if (typeof Chart !== 'undefined') {
            const quality = {
                labels:m.timeline.map(b=>new Date(b.started*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})),
                datasets:[
                    {label:'Ragas answer relevancy',data:m.timeline.map(b=>b.ragas_answer_relevancy ?? null),borderColor:'#7c7ff5'},
                    {label:'Ragas faithfulness',data:m.timeline.map(b=>b.ragas_faithfulness ?? null),borderColor:'#22c55e'}
                ]
            };
            if (qualityChart) {
                qualityChart.data=quality;
                qualityChart.update('none');
            } else qualityChart = new Chart(node('quality-chart'), {
                type:'line',data:quality,
                options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{color:'#a0a8c0'}}},
                    scales:{x:{ticks:{color:'#a0a8c0',maxTicksLimit:8}},y:{suggestedMin:0,suggestedMax:1,ticks:{color:'#a0a8c0'}}}}
            });
            const data = {
                labels: m.timeline.map(b=>new Date(b.started*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})),
                datasets: [
                    {label:'Requests',data:m.timeline.map(b=>b.requests),borderColor:'#7c7ff5',yAxisID:'volume',tension:.2},
                    {label:'Errors',data:m.timeline.map(b=>b.errors),borderColor:'#ef4444',yAxisID:'volume',tension:.2},
                    {label:'p95 seconds',data:m.timeline.map(b=>b.p95_ms == null ? null : b.p95_ms/1000),borderColor:'#22c55e',yAxisID:'latency',tension:.2},
                ]
            };
            if (trafficChart) {
                trafficChart.data=data;
                trafficChart.update('none');
            } else trafficChart = new Chart(node('traffic-chart'), {
                type:'line',data,
                options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{color:'#a0a8c0'}}},
                    scales:{x:{ticks:{color:'#a0a8c0',maxTicksLimit:8}},volume:{beginAtZero:true,ticks:{color:'#a0a8c0',precision:0}},
                    latency:{position:'right',beginAtZero:true,ticks:{color:'#22c55e'},grid:{drawOnChartArea:false}}}}
            });
        }
    } catch {
        node('fresh').textContent = 'Refresh failed. Reconnecting…';
    } finally {
        setTimeout(refresh, 5000);
    }
}
refresh();
