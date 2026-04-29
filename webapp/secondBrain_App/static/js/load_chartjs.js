// Chart.js CDN loader for histogram
(function() {
    if (!window.Chart) {
        var script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
        script.onload = function() {
            if (window.initFocusHistogram) window.initFocusHistogram();
        };
        document.head.appendChild(script);
    } else {
        if (window.initFocusHistogram) window.initFocusHistogram();
    }
})();
