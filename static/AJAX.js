// Definitions.
document.addEventListener('DOMContentLoaded', function() {
    const textForm = document.getElementById('textForm');
    const fileForm = document.getElementById('fileForm');
    const textResult = document.getElementById('textResult');
    const csvResult = document.getElementById('csvResult');
    const loadingDiv = document.getElementById('loading');
    let span_color;

    // Function to pick color of result's border based on sentiment.
    function pickColor(avgSentiment) {
        let color;
        if (avgSentiment === 'negative') {
            color = "#ff002f"
        } else if (avgSentiment === 'mostly negative') {
            color = "#ff51e9"
        } else if (avgSentiment === 'neutral') {
            color = "#2941ff"
        } else if (avgSentiment === 'mostly positive') {
            color = "#00C3FF"
        } else {
            color = "#00FF2C"
        }
        return color
    }

    // Function to ensure text inputs meet the 5 word minimum.
    function meetsCharCount(){
        input = document.getElementById("textinput").value;
        return input.length > 5;
    }

    // Function to handle response data from the server.
    function handleResponse(data) {
        // Display result for text input.
        if (data.input_text) {
            span_color = pickColor(data.predicted_rating_text)
            textResult.innerHTML = `
                <h2>Result</h2>
                <p>This text is <span style="border-bottom: dotted 2px ${span_color}">${data.predicted_rating_text}</span>.</p>
            `;
            textResult.classList.remove('hidden');
            textResult.classList.add('show');
            setTimeout(() => {
                textResult.classList.remove('show');
                textResult.classList.add('hidden');
            }, 10000);
        }

        // Display result for CSV upload.
        if (data.avg_sentiment) {
            span_color = pickColor(data.avg_sentiment)
            csvResult.innerHTML = `
                <h2>Result</h2>
                <p>This file contains a majority of <span style="border-bottom: dotted 2px ${span_color}">${data.avg_sentiment}</span> reviews.</p>
                <form class="bottom" action="/download_temp" method="get">
                    <input class="button big" type="submit" value="Download Report">
                </form>
            `;
            csvResult.classList.remove('hidden');
            csvResult.classList.add('show');
        }

        // Hide loading spinner when response is received.
        loadingDiv.classList.remove('show');
        loadingDiv.classList.add('hidden');
    }

    // Function to submit the form data and handle loading state.
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

    // Event listeners.
    // Text input -- rejects input if less than 5 characters.
    textForm.addEventListener('submit', function(event) {
        if (!meetsCharCount()) {
            event.preventDefault();
            textResult.innerHTML = `
                <h2>Error</h2>
                <p>Input too short; please enter at least 5 characters.</p>
            `;
            textResult.classList.remove('hidden');
            textResult.classList.add('show');
            setTimeout(() => {
                textResult.classList.remove('show');
                textResult.classList.add('hidden');
            }, 10000);
        } else {
            submitForm(textForm, event);
        }
    });

    // CSV upload.
    fileForm.addEventListener('submit', function(event) {
        submitForm(fileForm, event);
    });
});