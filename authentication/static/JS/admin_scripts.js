// soil_moisture_app/static/js/admin_scripts.js
document.addEventListener('DOMContentLoaded', function() {
    // Sidebar Toggle for Mobile
    const sidebarToggle = document.getElementById('sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function(e) {
            e.preventDefault()
            document.querySelector('.sidebar').classList.toggle('active');
        });
    }

    // Chart.js for Moisture Trends
    const moistureCtx = document.getElementById('moistureChart');
    if (moistureCtx) {
        // Get chart data from Django template variable
        const chartDataElement = document.getElementById('chartData');
        let chartData = null;
        
        if (chartDataElement) {
            try {
                chartData = JSON.parse(chartDataElement.textContent);
            } catch (e) {
                console.error('Error parsing chart data:', e);
            }
        }

        if (chartData && chartData.labels && chartData.data) {
            new Chart(moistureCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: chartData.labels,
                    datasets: [{
                        label: 'Average Soil Moisture (%)',
                        data: chartData.data,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: '#3498db',
                        pointBorderColor: '#ffffff',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Daily Average Soil Moisture Trends'
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            },
                            grid: {
                                display: true,
                                color: 'rgba(0,0,0,0.1)'
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Moisture (%)'
                            },
                            grid: {
                                display: true,
                                color: 'rgba(0,0,0,0.1)'
                            },
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    hover: {
                        animationDuration: 200
                    }
                }
            });
        } else {
            // Display message when no data is available
            const canvas = moistureCtx;
            const ctx = canvas.getContext('2d');
            ctx.font = '16px Arial';
            ctx.fillStyle = '#666';
            ctx.textAlign = 'center';
            ctx.fillText('No data available for chart', canvas.width / 2, canvas.height / 2);
        }
    }

    // Handle section navigation
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.content-section');

    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            
            // Hide all sections
            sections.forEach(section => {
                section.style.display = 'none';
            });
            
            // Show target section
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.style.display = 'block';
            }
            
            // Update active nav link
            navLinks.forEach(navLink => {
                navLink.classList.remove('active');
            });
            this.classList.add('active');
        });
    });

    // Show dashboard section by default
    const dashboardSection = document.getElementById('dashboard');
    if (dashboardSection) {
        dashboardSection.style.display = 'block';
    }
});

document.addEventListener('DOMContentLoaded', function() {
    let chart = null;

    // Sample data structure that matches your Django backend
    function generateSampleData(currentMoisture, temperature, humidity, location) {
        const data = [];
        const today = new Date();

        for (let i = 0; i < 7; i++) {
            const date = new Date(today);
            date.setDate(today.getDate() + i);

            // Simulate realistic moisture prediction variations
            const moistureVariation = (Math.random() - 0.5) * 10; // ±5% variation
            const tempVariation = (Math.random() - 0.5) * 6; // ±3°C variation
            const humidityVariation = (Math.random() - 0.5) * 20; // ±10% variation

            data.push({
                date: date.toISOString().split('T')[0],
                moisture: Math.max(0, Math.min(100, currentMoisture + moistureVariation)),
                temperature: temperature + tempVariation,
                humidity: Math.max(0, Math.min(100, humidity + humidityVariation))
            });
        }
        return data;
    }

    function updateForecastTable(forecastData) {
        const tbody = document.querySelector('#forecastTable tbody');
        tbody.innerHTML = '';

        forecastData.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.date}</td>
                <td>${row.moisture.toFixed(2)}</td>
                <td>${row.temperature.toFixed(2)}</td>
                <td>${row.humidity.toFixed(2)}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function createChart(forecastData) {
        const ctx = document.getElementById('soilMoistureChart').getContext('2d');

        // Destroy existing chart if it exists
        if (chart) {
            chart.destroy();
        }

        // Prepare chart data that matches your Django structure
        const chartData = {
            labels: forecastData.map(item => item.date),
            datasets: [
                {
                    label: 'Predicted Moisture (%)',
                    data: forecastData.map(item => item.moisture),
                    borderColor: '#1E90FF',
                    backgroundColor: 'rgba(30, 144, 255, 0.2)',
                    yAxisID: 'y1',
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Temperature (°C)',
                    data: forecastData.map(item => item.temperature),
                    borderColor: '#FF4500',
                    backgroundColor: 'rgba(255, 69, 0, 0.2)',
                    yAxisID: 'y2',
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Humidity (%)',
                    data: forecastData.map(item => item.humidity),
                    borderColor: '#32CD32',
                    backgroundColor: 'rgba(50, 205, 50, 0.2)',
                    yAxisID: 'y1',
                    tension: 0.4,
                    fill: false
                }
            ]
        };

        chart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: '7-Day Soil Moisture Forecast'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Moisture & Humidity (%)'
                        },
                        min: 0,
                        max: 100
                    },
                    y2: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Temperature (°C)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                interaction: {
                    mode: 'index',
                    intersect: false,
                }
            }
        });
    }

    function createMoistureChart() {
        const ctx = document.getElementById('soilMoistureChart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (chart) {
            chart.destroy();
        }
        
        // Get chart data from the Django template
        const chartDataScript = document.getElementById('chartData');
        if (!chartDataScript) {
            console.log('No chart data available - using sample data');
            const forecastData = generateSampleData(45, 25, 65, 'Kampala');
            createChart(forecastData);
            updateForecastTable(forecastData);
            return;
        }
        
        let chartData;
        try {
            chartData = JSON.parse(chartDataScript.textContent);
        } catch (e) {
            console.error('Error parsing chart data:', e);
            const forecastData = generateSampleData(45, 25, 65, 'Kampala');
            createChart(forecastData);
            updateForecastTable(forecastData);
            return;
        }
        
        // Create the chart with the data from Django
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData.labels,
                datasets: chartData.datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: '7-Day Soil Moisture Forecast'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Moisture & Humidity (%)'
                        },
                        min: 0,
                        max: 100
                    },
                    y2: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Temperature (°C)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                interaction: {
                    mode: 'index',
                    intersect: false,
                }
            }
        });
    }

    // Form submission handler
    document.getElementById('predictionForm')?.addEventListener('submit', function (e) {
        e.preventDefault();

        const formData = new FormData(this);
        const currentMoisture = parseFloat(formData.get('soil_moisture_percent'));
        const temperature = parseFloat(formData.get('temperature'));
        const humidity = parseFloat(formData.get('humidity'));
        const location = formData.get('location');

        // Generate sample forecast data
        const forecastData = generateSampleData(currentMoisture, temperature, humidity, location);

        // Update table and chart
        updateForecastTable(forecastData);
        createChart(forecastData);
    });

    // Initialize with either Django data or sample data
    createMoistureChart();
});