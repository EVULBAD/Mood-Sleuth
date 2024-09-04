document.addEventListener('DOMContentLoaded', function() {
    const textForm = document.getElementById('textForm');
    const fileForm = document.getElementById('fileForm');
    const textResult = document.getElementById('textResult');
    const csvResult = document.getElementById('csvResult');
    const loadingDiv = document.getElementById('loading');

    function handleResponse(data) {
        if (data.input_text) {
            textResult.innerHTML = `
                        <h2>Result</h2>
                        <p>Input Text: ${data.input_text}</p>
                        <p>This text appears to be ${data.predicted_rating_text}.</p>
                    `;
            textResult.classList.remove('hidden');
        }

        if (data.avg_sentiment) {
            csvResult.innerHTML = `
                        <h2>CSV Analysis Summary</h2>
                        <p>This file contains mostly ${data.avg_sentiment} reviews.</p>
                        <p>Here's a positive review: ${data.example_positive_review}</p>
                        <p>And here's a negative one: ${data.example_negative_review}</p>
                        <p>For a detailed report, please download and review the following csv:</p>
                        <form action="/download_predictions" method="post">
                            <input type="hidden" name="sentiment_analysis_report" value="${data.sentiment_analysis_report}">
                            <input type="submit" value="Download Predictions CSV">
                        </form>
                    `;
            csvResult.classList.remove('hidden');
        }

        loadingDiv.classList.add('hidden');
    }

    function submitForm(form, event) {
        event.preventDefault();
        loadingDiv.classList.remove('hidden');
        textResult.classList.add('hidden');
        csvResult.classList.add('hidden');

        const formData = new FormData(form);
        fetch('/', {
            method: 'POST',
            body: formData,
        })
            .then(response => response.json())
            .then(data => handleResponse(data))
            .catch(error => console.error('Error:', error));
    }

    textForm.addEventListener('submit', function(event) {
        submitForm(textForm, event);
    });

    fileForm.addEventListener('submit', function(event) {
        submitForm(fileForm, event);
    });
});