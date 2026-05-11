// Levenshtein distance implementation
function levenshteinDistance(a, b) {
    const matrix = [];
    for (let i = 0; i <= b.length; i++) {
        matrix[i] = [i];
    }
    for (let j = 0; j <= a.length; j++) {
        matrix[0][j] = j;
    }
    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            if (b.charAt(i - 1) === a.charAt(j - 1)) {
                matrix[i][j] = matrix[i - 1][j - 1];
            } else {
                matrix[i][j] = Math.min(
                    matrix[i - 1][j - 1] + 1,
                    matrix[i][j - 1] + 1,
                    matrix[i - 1][j] + 1
                );
            }
        }
    }
    return matrix[b.length][a.length];
}

function calculateCER(reference, predicted) {
    const dist = levenshteinDistance(predicted, reference);
    return (dist / reference.length) * 100;
}

function calculateWER(reference, predicted) {
    const refWords = reference.split(/\s+/).filter(w => w.length > 0);
    const predWords = predicted.split(/\s+/).filter(w => w.length > 0);
    const dist = levenshteinDistance(predWords.join(' '), refWords.join(' '));
    return (dist / refWords.length) * 100;
}

// Update file labels when files are selected
document.getElementById('gtFiles').addEventListener('change', (e) => {
    const label = document.getElementById('gtFilesLabel');
    const wrapper = e.target.parentElement.querySelector('.file-input-label');
    if (e.target.files.length > 0) {
        label.textContent = `✓ ${e.target.files.length} file(s) selected`;
        wrapper.classList.add('has-files');
    } else {
        label.textContent = 'Click to select GT files...';
        wrapper.classList.remove('has-files');
    }
});

document.getElementById('predFiles').addEventListener('change', (e) => {
    const label = document.getElementById('predFilesLabel');
    const wrapper = e.target.parentElement.querySelector('.file-input-label');
    if (e.target.files.length > 0) {
        label.textContent = `✓ ${e.target.files.length} file(s) selected`;
        wrapper.classList.add('has-files');
    } else {
        label.textContent = 'Click to select prediction files...';
        wrapper.classList.remove('has-files');
    }
});

document.getElementById('calculateBtn').addEventListener('click', async () => {
    const gtFiles = document.getElementById('gtFiles').files;
    const predFiles = document.getElementById('predFiles').files;
    const btn = document.getElementById('calculateBtn');

    if (gtFiles.length === 0 || predFiles.length === 0) {
        alert('Please select files for both GT and predictions!');
        return;
    }

    if (gtFiles.length !== predFiles.length) {
        alert('Number of GT and prediction files must match!');
        return;
    }

    btn.disabled = true;
    btn.innerHTML = '<span class="icon"><i class="fas fa-spinner fa-spin"></i></span><span>Processing...</span>';
    document.getElementById('resultsCard').style.display = 'block';
    document.getElementById('chartCard').style.display = 'block';
    document.getElementById('results').innerHTML = '<p class="has-text-grey"><em>⏳ Calculating metrics...</em></p>';

    const pageMetrics = [];
    let totalCER_CS = 0, totalWER_CS = 0, totalCER_CI = 0, totalWER_CI = 0;
    let totalChars = 0, totalWords = 0;

    for (let i = 0; i < gtFiles.length; i++) {
        const gtText = await gtFiles[i].text();
        const predText = await predFiles[i].text();

        // Case-sensitive
        const cer_cs = calculateCER(gtText, predText);
        const wer_cs = calculateWER(gtText, predText);

        // Case-insensitive
        const gtLower = gtText.toLowerCase();
        const predLower = predText.toLowerCase();
        const cer_ci = calculateCER(gtLower, predLower);
        const wer_ci = calculateWER(gtLower, predLower);

        pageMetrics.push({
            page: gtFiles[i].name.replace('.txt', ''),
            cer_case_sensitive: cer_cs,
            wer_case_sensitive: wer_cs,
            cer_case_insensitive: cer_ci,
            wer_case_insensitive: wer_ci
        });

        totalCER_CS += cer_cs * gtText.length;
        totalWER_CS += wer_cs * gtText.split(/\s+/).filter(w => w.length > 0).length;
        totalCER_CI += cer_ci * gtLower.length;
        totalWER_CI += wer_ci * gtLower.split(/\s+/).filter(w => w.length > 0).length;
        totalChars += gtText.length;
        totalWords += gtText.split(/\s+/).filter(w => w.length > 0).length;
    }

    const avgCER_CS = totalChars > 0 ? (totalCER_CS / totalChars) : 0;
    const avgWER_CS = totalWords > 0 ? (totalWER_CS / totalWords) : 0;
    const avgCER_CI = totalChars > 0 ? (totalCER_CI / totalChars) : 0;
    const avgWER_CI = totalWords > 0 ? (totalWER_CI / totalWords) : 0;

    // Display results
    let resultsHTML = '';
    
    // Overall metrics cards
    resultsHTML += '<div class="columns is-multiline mb-5">';
    resultsHTML += '<div class="column is-3"><div class="metric-box"><div class="metric-label">CER (Case-Sensitive)</div><div class="metric-value">' + avgCER_CS.toFixed(2) + '%</div></div></div>';
    resultsHTML += '<div class="column is-3"><div class="metric-box"><div class="metric-label">WER (Case-Sensitive)</div><div class="metric-value">' + avgWER_CS.toFixed(2) + '%</div></div></div>';
    resultsHTML += '<div class="column is-3"><div class="metric-box"><div class="metric-label">CER (Case-Insensitive)</div><div class="metric-value">' + avgCER_CI.toFixed(2) + '%</div></div></div>';
    resultsHTML += '<div class="column is-3"><div class="metric-box"><div class="metric-label">WER (Case-Insensitive)</div><div class="metric-value">' + avgWER_CI.toFixed(2) + '%</div></div></div>';
    resultsHTML += '</div>';
    
    // Per-page metrics table
    resultsHTML += '<h3 class="title is-5">Per-Page Metrics</h3>';
    resultsHTML += '<div class="table-container"><table class="table is-striped is-hoverable is-fullwidth">';
    resultsHTML += '<thead><tr><th>Page</th><th class="has-text-right">CER CS (%)</th><th class="has-text-right">WER CS (%)</th><th class="has-text-right">CER CI (%)</th><th class="has-text-right">WER CI (%)</th></tr></thead><tbody>';
    pageMetrics.forEach(m => {
        resultsHTML += `<tr><td><strong>${m.page}</strong></td><td class="has-text-right">${m.cer_case_sensitive.toFixed(2)}</td><td class="has-text-right">${m.wer_case_sensitive.toFixed(2)}</td><td class="has-text-right">${m.cer_case_insensitive.toFixed(2)}</td><td class="has-text-right">${m.wer_case_insensitive.toFixed(2)}</td></tr>`;
    });
    resultsHTML += '</tbody></table></div>';
    document.getElementById('results').innerHTML = resultsHTML;

    // Chart
    const ctx = document.getElementById('chart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: pageMetrics.map(m => m.page),
            datasets: [{
                label: 'CER (Case-Sensitive)',
                data: pageMetrics.map(m => m.cer_case_sensitive),
                borderColor: '#FF6B6B',
                backgroundColor: 'rgba(255, 107, 107, 0.2)',
                fill: false,
                pointStyle: 'circle'
            }, {
                label: 'WER (Case-Sensitive)',
                data: pageMetrics.map(m => m.wer_case_sensitive),
                borderColor: '#4ECDC4',
                backgroundColor: 'rgba(78, 205, 196, 0.2)',
                fill: false,
                pointStyle: 'rect'
            }, {
                label: 'CER (Case-Insensitive)',
                data: pageMetrics.map(m => m.cer_case_insensitive),
                borderColor: '#FFE66D',
                backgroundColor: 'rgba(255, 230, 109, 0.2)',
                fill: false,
                pointStyle: 'triangle'
            }, {
                label: 'WER (Case-Insensitive)',
                data: pageMetrics.map(m => m.wer_case_insensitive),
                borderColor: '#95E1D3',
                backgroundColor: 'rgba(149, 225, 211, 0.2)',
                fill: false,
                pointStyle: 'rectRot'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'CER/WER per Page - Case-Sensitive vs Case-Insensitive'
                }
            },
            scales: {
                x: { title: { display: true, text: 'Page Number' } },
                y: { title: { display: true, text: 'Error Rate (%)' } }
            }
        }
    });

    btn.disabled = false;
    btn.innerHTML = '<span class="icon">📊</span><span>Calculate Metrics</span>';
});