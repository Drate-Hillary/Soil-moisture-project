// soil_moisture_app/static/js/admin_scripts.js
document.addEventListener('DOMContentLoaded', function() {
    // Sidebar Toggle for Mobile
    const sidebarToggle = document.getElementById('sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            document.querySelector('.sidebar').classList.toggle('active');
        });
    }

    // Chart.js for Moisture Trends
    const moistureCtx = document.getElementById('moistureChart');
    if (moistureCtx) {
        new Chart(moistureCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04'],
                datasets: [{
                    label: 'Soil Moisture (%)',
                    data: [40, 42, 45, 47],
                    borderColor: '#3498db',
                    fill: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'Date' } },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: 'Moisture (%)' }
                    }
                }
            }
        });
    }

    // Chart.js for Prediction Chart
    const predictionCtx = document.getElementById('predictionChart');
    if (predictionCtx) {
        const predictions = JSON.parse('{{ predictions|safe }}'); // Pass predictions as JSON from context
        new Chart(predictionCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: predictions.map(p => new Date(p.timestamp).toLocaleString()),
                datasets: [{
                    label: 'Predicted Moisture (%)',
                    data: predictions.map(p => p.predicted_moisture),
                    borderColor: 'rgba(75, 192, 192, 1)',
                    fill: false
                }, {
                    label: 'Input Moisture (%)',
                    data: predictions.map(p => p.input_moisture),
                    borderColor: 'rgba(255, 99, 132, 1)',
                    fill: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'Timestamp' } },
                    y: { title: { display: true, text: 'Moisture (%)' } }
                }
            }
        });
    }
});