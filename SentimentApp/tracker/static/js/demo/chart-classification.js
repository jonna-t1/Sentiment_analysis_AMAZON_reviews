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

var arrayLength = dates.length;
var formattedDates = []
for (var i = 0; i < arrayLength; i++) {
  var dateSplit = dates[i].split(" ");
  // console.log(dateSplit[0]);
  formattedDates.push(dateSplit[0])
}

var posRow = [];
var pos_precision = [];
var pos_recall = [];
var pos_f1 = [];
var pos_support = [];

var negRow = [];
var neg_precision = [];
var neg_recall = [];
var neg_f1 = [];
var neg_support = [];

var classRow = [];
var precision = [];
var recall = [];
var f1 = [];
var support = [];

for (var key in classObjects) {
    classRow = classObjects[key];
    precision.push(classRow[0]);
    recall.push(classRow[1]);
    f1.push(classRow[2]);
    support.push(classRow[3]);
    // precision.append(classRow[0])
}
for (var key in posObjects) {
    posRow = posObjects[key];
    pos_precision.push(posRowRow[0]);
    pos_recall.push(posRowRow[1]);
    pos_f1.push(posRowRow[2]);
    pos_support.push(posRowRow[3]);
    // precision.append(classRow[0])
}
for (var key in negObjects) {
    negRow = negObjects[key];
    neg_precision.push(negRow[0]);
    neg_recall.push(negRow[1]);
    neg_f1.push(negRow[2]);
    neg_support.push(negRow[3]);
    // precision.append(classRow[0])
}


if (neg_f1.length != formattedDates.length){
    alert("Incorrect array sizes dont match: Contact admin")
    // document.getElementById("myLineChart").innerHTML = err.message;
}


var config = {

  type: 'line',
  data: {
    labels: formattedDates,
    datasets: [
    {
      // weighted avg dataset
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
      // data: [0, 10000, 5000, 15000, 10000, 20000, 15000, 25000, 20000, 30000, 25000, 40000],
      data: [1,2],
    },{
      // positive scores
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
      // data: [0, 10000, 5000, 15000, 10000, 20000, 15000, 25000, 20000, 30000, 25000, 40000],
      data: [1,2],
    },{
      // negative scores
      label: "Line C: ",
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
      // data: [0, 10000, 5000, 15000, 10000, 20000, 15000, 25000, 20000, 30000, 25000, 40000],
      data: [1,2],
    },
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
          // var datasetIndex = chart.datasets[tooltipItem.datasetIndex] || '';
          printIndex(tooltipItem.index);
          // console.log(tooltipItem.yLabel)
          return datasetLabel + ' ' + tooltipItem.yLabel;
        }
      }
    }
  }
}

config.data.datasets[0].data = precision;
config.data.datasets[1].data = pos_precision;
config.data.datasets[2].data = neg_precision;
myLineChart = new Chart(ctx, config);


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

