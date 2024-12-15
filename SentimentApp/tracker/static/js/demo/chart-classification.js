// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

function number_format(number, decimals, dec_point, thousands_sep) {
  // *     example: number_format(1234.56, 2, ',', ' ');
  // *     return: '1 234,56'
  number = (number + '').replace(',', '').replace(' ', '');
  var n = !isFinite(+number) ? 0 : +number,
    prec = !isFinite(+decimals) ? 0 : Math.abs(decimals),
    sep = (typeof thousands_sep === 'undefined') ? ',' : thousands_sep,
    dec = (typeof dec_point === 'undefined') ? '.' : dec_point,
    s = '',
    toFixedFix = function(n, prec) {
      var k = Math.pow(10, prec);
      return '' + Math.round(n * k) / k;
    };
  // Fix for IE parseFloat(0.55).toFixed(0) = 0;
  s = (prec ? toFixedFix(n, prec) : '' + Math.round(n)).split('.');
  if (s[0].length > 3) {
    s[0] = s[0].replace(/\B(?=(?:\d{3})+(?!\d))/g, sep);
  }
  if ((s[1] || '').length < prec) {
    s[1] = s[1] || '';
    s[1] += new Array(prec - s[1].length + 1).join('0');
  }
  return s.join(dec);
}


var precisionDIV = document.getElementById('precision');
var recallDIV = document.getElementById('recall');
var f1DIV = document.getElementById('f1');

// Area Chart Example
var ctx = document.getElementById("myAreaChart");

// Filtered data arrays
var formattedDates = [];
var pos_precision = [];
var neg_precision = [];
var precision = [];

for (var i = 0; i < dates.length; i++) {
  var dateSplit = dates[i].split(" ");
  var date = dateSplit[0];

  // Check if precision data for the date is available, and filter accordingly
  if (pos_precision[i] !== null && pos_precision[i] !== undefined &&
      neg_precision[i] !== null && neg_precision[i] !== undefined &&
      precision[i] !== null && precision[i] !== undefined) {

    formattedDates.push(date);
    pos_precision.push(pos_precision[i]);
    neg_precision.push(neg_precision[i]);
    precision.push(precision[i]);
  }
}

var config = {
  type: 'line',
  data: {
    labels: formattedDates,
    datasets: [
      {
        // Avg precision line
        label: "Avg line: ",
        lineTension: 0.3,
        backgroundColor: "rgba(78, 115, 223, 0.05)",
        borderColor: "rgba(78, 115, 223, 1)",
        pointRadius: 3,
        pointBackgroundColor: "rgba(78, 115, 223, 1)",
        pointBorderColor: "rgba(78, 115, 223, 1)",
        pointHoverRadius: 3,
        pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",
        pointHoverBorderColor: "rgba(78, 115, 223, 1)",
        pointHitRadius: 10,
        pointBorderWidth: 2,
        data: precision, // Filtered precision data
      },
      {
        // Positive precision
        label: "Pos Line: ",
        lineTension: 0.3,
        backgroundColor: "rgba(78, 115, 223, 0.05)",
        borderColor: "rgba(78, 115, 223, 1)",
        pointRadius: 3,
        pointBackgroundColor: "rgba(78, 115, 223, 1)",
        pointBorderColor: "rgba(78, 115, 223, 1)",
        pointHoverRadius: 3,
        pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",
        pointHoverBorderColor: "rgba(78, 115, 223, 1)",
        pointHitRadius: 10,
        pointBorderWidth: 2,
        data: pos_precision, // Filtered positive precision data
      },
      {
        // Negative precision
        label: "Neg Line: ",
        lineTension: 0.3,
        backgroundColor: "rgba(78, 115, 223, 0.05)",
        borderColor: "rgba(78, 115, 223, 1)",
        pointRadius: 3,
        pointBackgroundColor: "rgba(78, 115, 223, 1)",
        pointBorderColor: "rgba(78, 115, 223, 1)",
        pointHoverRadius: 3,
        pointHoverBackgroundColor: "rgba(78, 115, 223, 1)",
        pointHoverBorderColor: "rgba(78, 115, 223, 1)",
        pointHitRadius: 10,
        pointBorderWidth: 2,
        data: neg_precision, // Filtered negative precision data
      }
    ],
  },
  options: {
    onClick: graphClickEvent,
    maintainAspectRatio: false,
    layout: {
      padding: {
        left: 10,
        right: 25,
        top: 25,
        bottom: 0
      }
    },
    scales: {
      xAxes: [{
        time: {
          unit: 'date'
        },
        gridLines: {
          display: false,
          drawBorder: false
        },
        ticks: {
          maxTicksLimit: 7
        }
      }],
      yAxes: [{
        ticks: {
          maxTicksLimit: 5,
          padding: 10,
        },
        gridLines: {
          color: "rgb(234, 236, 244)",
          zeroLineColor: "rgb(234, 236, 244)",
          drawBorder: false,
          borderDash: [2],
          zeroLineBorderDash: [2]
        }
      }],
    },
    legend: {
      display: false
    },
    tooltips: {
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      titleMarginBottom: 10,
      titleFontColor: '#6e707e',
      titleFontSize: 14,
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: false,
      intersect: false,
      mode: 'index',
      caretPadding: 10,
      callbacks: {
        label: function(tooltipItem, chart) {
          var datasetLabel = chart.datasets[tooltipItem.datasetIndex].label || '';
          return datasetLabel + ' ' + tooltipItem.yLabel;
        }
      }
    }
  }
}

// Initialize the chart
myLineChart = new Chart(ctx, config);

// Add event listeners for switching data views
precisionDIV.style.cursor = 'pointer';
precisionDIV.onclick = function() {
    myLineChart.config.data.datasets[0].data = precision;
    myLineChart.config.data.datasets[1].data = pos_precision;
    myLineChart.config.data.datasets[2].data = neg_precision;
    myLineChart.update();
};

recallDIV.style.cursor = 'pointer';
recallDIV.onclick = function() {
    myLineChart.config.data.datasets[0].data = recall;
    myLineChart.config.data.datasets[1].data = pos_recall;
    myLineChart.config.data.datasets[2].data = neg_recall;
    myLineChart.update();
};

f1DIV.style.cursor = 'pointer';
f1DIV.onclick = function() {
    myLineChart.config.data.datasets[0].data = f1;
    myLineChart.config.data.datasets[1].data = pos_f1;
    myLineChart.config.data.datasets[2].data = neg_f1;
    myLineChart.update();
};
