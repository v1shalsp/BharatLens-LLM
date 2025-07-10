// app.js (Corrected for On-Demand Analysis)

document.addEventListener('DOMContentLoaded', function() {
    const analysisForm = document.getElementById('analysisForm');
    const topicInput = document.getElementById('topicInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');

    // Form submission handler for on-demand analysis
    analysisForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const topic = topicInput.value.trim();
        if (!topic) return;

        // Show loading indicator and hide previous results
        loadingIndicator.style.display = 'block';
        resultsSection.style.display = 'none';
        analyzeBtn.disabled = true;

        try {
            // --- FIX: The URL must be '/api/analyze' to match the Python server ---
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ topic: topic })
            });

            const data = await response.json();

            if (response.ok) {
                // If successful, display the results immediately
                displayResults(data);
            } else {
                // If the server returns an error, show it
                showError(data.error || 'Analysis failed. Please try again.');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            showError('Network error. Please check your connection and try again.');
        } finally {
            // Hide loading indicator and re-enable the button
            loadingIndicator.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });

    // Function to display the analysis results on the page
    function displayResults(data) {
        window.lastAnalysisData = data;
        document.getElementById('resultsHeader').innerText = `Analysis for: ${data.topic}`;

        const perspectivesCards = document.getElementById('perspectivesCards');
        perspectivesCards.innerHTML = '';
        if (data.perspectives && data.perspectives.length > 0) {
            document.getElementById('perspectivesSection').style.display = 'block';
            data.perspectives.forEach((p, idx) => {
                const card = document.createElement('div');
                card.className = 'col-md-6 mb-3';
                card.innerHTML = `<div class="card h-100 shadow-sm"><div class="card-body"><h5 class="card-title">${p.label}</h5><p class="card-text">${formatText(p.summary)}</p>` +
                    (p.evidence && p.evidence.length > 0 ? `<button class="btn btn-link btn-sm show-evidence-btn" data-idx="${idx}"><i class="fas fa-search"></i> Show Evidence</button><div class="evidence-section" id="evidence-${idx}" style="display:none;"></div>` : '') +
                    `</div></div>`;
                perspectivesCards.appendChild(card);
            });
            // Add event listeners for show evidence buttons
            document.querySelectorAll('.show-evidence-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const idx = this.getAttribute('data-idx');
                    const section = document.getElementById(`evidence-${idx}`);
                    if (section.style.display === 'none') {
                        section.innerHTML = `<ul class="evidence-list">${data.perspectives[idx].evidence.map(e => `<li>${formatText(e)}</li>`).join('')}</ul>`;
                        section.style.display = 'block';
                        this.innerHTML = '<i class="fas fa-eye-slash"></i> Hide Evidence';
                    } else {
                        section.style.display = 'none';
                        this.innerHTML = '<i class="fas fa-search"></i> Show Evidence';
                    }
                });
            });
        }

        document.getElementById('executiveSummary').innerHTML = formatText(data.executive_summary);
        document.getElementById('biasReport').innerHTML = formatText(data.detailed_bias_report);

        const metricsDiv = document.getElementById('evaluationMetrics');
        metricsDiv.innerHTML = '';
        if (data.summary_evaluation) {
            Object.entries(data.summary_evaluation).forEach(([metric, score]) => {
                const metricDiv = document.createElement('div');
                metricDiv.className = 'evaluation-item';
                metricDiv.innerHTML = `<div class="evaluation-score">${score}</div><div>${metric}</div>`;
                metricsDiv.appendChild(metricDiv);
            });
        }

        // Graphs: Source Bias, Sentiment Trend, Stance Heatmap (if present)
        if (data.visualizations) {
            // Interactive stance chart with Plotly
            if (data.visualizations.stance_dist_chart_data) {
                renderStancePlotlyChart(data.visualizations.stance_dist_chart_data);
                document.getElementById('sourceChartContainer').style.display = 'block';
            } else {
                updateChart('sourceChart', 'sourceChartContainer', data.visualizations.source_bias_chart_url, 'Source Bias Chart');
            }
            updateChart('historicalChart', 'historicalChartContainer', data.visualizations.sentiment_trend_chart_url, 'Sentiment Trend Over Time');
            updateChart('stanceHeatmap', 'stanceHeatmapContainer', data.visualizations.stance_sentiment_heatmap_url, 'Stance Sentiment Heatmap');
        }

        // --- Entities Section ---
        const entitiesSection = document.getElementById('entitiesSection');
        const entitiesList = document.getElementById('entitiesList');
        entitiesList.innerHTML = '';
        if (data.entities && data.entities.length > 0) {
            entitiesSection.style.display = 'block';
            data.entities.slice(0, 30).forEach(entity => {
                const tag = document.createElement('span');
                tag.className = 'entity-tag';
                tag.innerText = `${entity.text} (${entity.type}, ${entity.count})`;
                entitiesList.appendChild(tag);
            });
        } else {
            entitiesSection.style.display = 'none';
        }

        // --- Timeline Section ---
        const timelineSection = document.getElementById('timelineSection');
        const timelineList = document.getElementById('timelineList');
        timelineList.innerHTML = '';
        if (data.timeline && data.timeline.length > 0) {
            timelineSection.style.display = 'block';
            data.timeline.forEach(event => {
                const item = document.createElement('div');
                item.className = 'timeline-item';
                item.innerHTML = `<div class="timeline-date">${event.date ? event.date.split('T')[0] : ''}</div><div class="timeline-event">${formatText(event.event)}</div><div class="timeline-source"><a href="${event.source}" target="_blank">Source</a></div>`;
                timelineList.appendChild(item);
            });
        } else {
            timelineSection.style.display = 'none';
        }

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    function updateChart(imgId, containerId, url, altText) {
        const imgDiv = document.getElementById(imgId);
        const containerDiv = document.getElementById(containerId);
        if (url && imgDiv && containerDiv) {
            imgDiv.innerHTML = `<img src="${url}?t=${new Date().getTime()}" alt="${altText}" class="img-fluid">`;
            containerDiv.style.display = 'block';
        } else if (containerDiv) {
            containerDiv.style.display = 'none';
        }
    }

    function formatText(text) {
        if (!text) return '<p>No data available.</p>';
        return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
    }

    function showError(message) {
        resultsSection.innerHTML = `<div class="alert alert-danger"><i class="fas fa-exclamation-circle"></i> ${message}</div>`;
        resultsSection.style.display = 'block';
    }

    // Add Plotly stance chart rendering
    function renderStancePlotlyChart(chartData) {
        const chartDiv = document.getElementById('sourceChart');
        if (!chartData) {
            chartDiv.innerHTML = '<p>No data available.</p>';
            return;
        }
        const tracePro = { x: chartData.labels, y: chartData.pro, name: 'Pro', type: 'bar' };
        const traceAnti = { x: chartData.labels, y: chartData.anti, name: 'Anti', type: 'bar' };
        const traceNeutral = { x: chartData.labels, y: chartData.neutral, name: 'Neutral', type: 'bar' };
        const data = [tracePro, traceAnti, traceNeutral];
        const layout = { barmode: 'stack', title: 'Stance Distribution by Source', xaxis: { title: 'Source' }, yaxis: { title: 'Count' } };
        Plotly.newPlot(chartDiv, data, layout, {responsive: true});
    }
});