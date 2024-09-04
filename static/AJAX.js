document.addEventListener('DOMContentLoaded', function() {
    const textForm = document.getElementById('textForm');
    const fileForm = document.getElementById('fileForm');
    const textResult = document.getElementById('textResult');
    const csvResult = document.getElementById('csvResult');
    const loadingDiv = document.getElementById('loading');

    function pickColor(avgSentiment) {
        if (avgSentiment === 'negative') {
            span_color = "#ff002f"
        } else if (avgSentiment === 'mostly negative') {
            span_color = "#ff51e9"
        } else if (avgSentiment === 'neutral') {
            span_color = "#2941ff"
        } else if (avgSentiment === 'mostly positive') {
            span_color = "#00C3FF"
        } else {
            span_color = "#00FF2C"
        }

        return span_color
    }

    function handleResponse(data) {
        if (data.input_text) {
            span = pickColor(data.predicted_rating_text)
            textResult.innerHTML = `
                <h2>Result</h2>
                <p>This text is <span style="border-bottom: dotted 2px ${span_color}">${data.predicted_rating_text}</span>.</p>
            `;
            textResult.classList.remove('hidden');
            textResult.classList.add('show');
            setTimeout(() => {
                textResult.classList.remove('show');
                textResult.classList.add('hidden');
            }, 7000);
        }

        if (data.avg_sentiment) {
            span = pickColor(data.avg_sentiment)
            csvResult.innerHTML = `
                <h2>Result</h2>
                <p>This file contains a majority of <span style="border-bottom: dotted 2px ${span_color}">${data.avg_sentiment}</span> reviews.</p>
                <form class="bottom" action="/download_temp" method="get">
                    <input type="submit" value="Download Full Report">
                </form>
            `;
            csvResult.classList.remove('hidden');
            csvResult.classList.add('show');
        }

        loadingDiv.classList.remove('show');
        loadingDiv.classList.add('hidden');
    }

    function submitForm(form, event) {
        event.preventDefault();
        loadingDiv.classList.remove('hidden');
        loadingDiv.classList.add('show');
        textResult.classList.remove('show');
        textResult.classList.add('hidden');
        csvResult.classList.remove('show');
        csvResult.classList.add('hidden');

        const formData = new FormData(form);
        fetch('/', {
            method: 'POST',
            body: formData,
        })
            .then(response => response.json())
            .then(data => handleResponse(data))
            .catch(error => {
                console.error('Error:', error);
                loadingDiv.classList.remove('show');
                loadingDiv.classList.add('hidden');
            });
    }

    textForm.addEventListener('submit', function(event) {
        submitForm(textForm, event);
    });

    fileForm.addEventListener('submit', function(event) {
        submitForm(fileForm, event);
    });
});