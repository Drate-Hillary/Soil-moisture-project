// Sidebar Toggle for Mobile
document.getElementById('sidebarToggle').addEventListener('click', function() {
    document.querySelector('.sidebar').classList.toggle('active');
});

// Sample Chart.js for Moisture Trends
const moistureCtx = document.getElementById('moistureChart').getContext('2d');
new Chart(moistureCtx, {
    type: 'line',
    data: {
        labels: ['2025-07-08 08:00', '2025-07-08 09:00', '2025-07-08 10:00'],
        datasets: [{
            label: 'Soil Moisture (%)',
            data: [40, 42, 45],
            borderColor: '#3498db',
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
    type: 'bar',
    data: {
        labels: ['Actual', 'Predicted'],
        datasets: [{
            label: 'Moisture (%)',
            data: [45, 47],
            backgroundColor: ['#3498db', '#e74c3c']
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