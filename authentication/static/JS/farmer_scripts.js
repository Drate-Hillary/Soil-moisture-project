// Sidebar Toggle for Mobile
document.getElementById('sidebarToggle').addEventListener('click', function() {
    document.querySelector('.sidebar').classList.toggle('active');
});

// Sample Chart.js for Moisture Trends
const moistureCtx = document.getElementById('moistureChart').getContext('2d');
new Chart(moistureCtx, {
    type: 'line',
    data: {
        labels: ['2025-07-08', '2025-07-07', '2025-07-06'],
        datasets: [{
            label: 'Moisture (%)',
            data: [45, 43, 40],
            borderColor: '#27ae60',
            fill: false
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    }
});

// Sample Chart.js for Predictions
const predictionCtx = document.getElementById('predictionChart').getContext('2d');
new Chart(predictionCtx, {
    type: 'line',
    data: {
        labels: ['Today', 'Tomorrow'],
        datasets: [{
            label: 'Predicted Moisture (%)',
            data: [45, 47],
            borderColor: '#e74c3c',
            fill: false
        }]
    },
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    }
});

// Sample Chart.js for Moisture Gauge
const gaugeCtx = document.getElementById('moistureGauge').getContext('2d');
new Chart(gaugeCtx, {
    type: 'doughnut',
    data: {
        datasets: [{
            data: [45, 55],
            backgroundColor: ['#27ae60', '#ecf0f1']
        }]
    },
    options: {
        responsive: true,
        circumference: 180,
        rotation: -90,
        cutout: '80%'
    }
});