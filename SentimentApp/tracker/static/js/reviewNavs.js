var precisionDIV = document.getElementById('precision');
var recallDIV = document.getElementById('recall');
var f1DIV = document.getElementById('f1');
precisionDIV.style.cursor = 'pointer';
precisionDIV.onclick = function() {
    window.location.href = 'http://localhost:8080/dataView/';
};
recallDIV.style.cursor = 'pointer';
recallDIV.onclick = function() {
    window.location.href = 'http://localhost:8080/dataView/matches/';
};
f1DIV.style.cursor = 'pointer';
f1DIV.onclick = function() {
    window.location.href = 'http://localhost:8080/dataView/falsematches/';
};
