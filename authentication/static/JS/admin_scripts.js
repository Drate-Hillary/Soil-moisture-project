// Sidebar Toggle for Mobile
document.getElementById('sidebarToggle').addEventListener('click', function() {
    document.querySelector('.sidebar').classList.toggle('active');
});

// Sample Chart.js for Moisture Trends
const ctx = document.getElementById('moistureChart').getContext('2d');
new Chart(ctx, {
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
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    }
});