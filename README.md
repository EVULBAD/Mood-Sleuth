# Summary
Mood Sleuth addresses the challenges of inefficiency and inaccuracy in manual sentiment analysis of customer feedback, a task that can be overwhelming for organizations handling large volumes of data. By automating the sentiment analysis process, Mood Sleuth enables businesses to categorize customer reviews into easily interpretable categories; not only streamlining the review process, but also eliminating the potential for human error and bias. The application’s use of machine learning, particularly logistic regression, allows for accurate sentiment prediction based on patterns in language and word usage, ensuring businesses can trust the insights it provides.

By offering both text input and CSV upload options, Mood Sleuth becomes a flexible, scalable solution for businesses of all sizes, allowing organizations to quickly analyze large sets of customer feedback – leading to faster, more informed decision-making. In an era where customer satisfaction is vital for competitive success, Mood Sleuth empowers businesses to stay responsive to their customers’ needs, driving improvements and fostering better customer relationships based on real-time feedback analysis.

# User Guide
_Note 1_: Mood Sleuth, having been trained on actual user-generated reviews, does best with inputs that reflect that. As such, copied reviews from ecommerce websites is the best way to see the most accurate results possible. Additionally, the application is less adept at categorizing text inputs containing few words and/or non-dictionary English (including typos, slang, emoticons, etc), so analyzing content with either of those factors present is not recommended.

_Note 2_: For CSV analysis, the first column of the CSV must be labeled "Review". Any columns aside from the "Review" column will be ignored.

**Sample Text Inputs**

_Positive_

&emsp;This lid is an outstanding piece of hardware. The lid snaps on SUPER TIGHT, in fact, I’d hate to try and pry it off. Tip: Put the lid on and turn the bucket over on a hard surface. Press on the bottom and it should snap on. The lid itself is robust enough to sit on, and seals the bucket easy. Takes up very little room of the original opening. This is really really really well made.

_Neutral_

&emsp;Works as a flash drive, not much for maintenance or performance.

_Negative_

&emsp;The leg tubes don’t fit into the clamps to lock the bed when it’s open. It could fold up when the pet steps up onto it. I couldn’t submit my review without giving a star!

**Sample CSVs**

Sample CSVs are available in the repository's "Sample CSVs" directory. They were not used to train Mood Sleuth and do not contain ratings, intended only for testing analysis capabilities. "reviews_testing.csv" is intentionally short in order to decrease processing time while debugging, while the other CSVs are their full lengths to ensure the application can handle them.

### Guide
1. Navigate to EVULBAD.com/mood-sleuth.

**Analyzing a text input:**

2. In the text box, input your text to be analyzed (5 characters minimum).
3. Click the "analyze" button immediately below the text box.
4. The text input will be processed, then the analysis will reveal itself in a box labeled
"result". This analysis will disappear after 10 seconds.

**Analyzing a CSV:**

5. Click the "upload CSV" button.
6. Navigate through your file system and pick a CSV to analyze. Once a file is successfully
selected, the "upload CSV" button will change color and read "uploaded!".
7. Click the "analyze" button immediately next to the "uploaded!" button.
8. The CSV input will be processed, then the average analysis of the CSV as a whole will
reveal itself in a box labeled "result", alongside a "download report" button.
9. Click the "download report" button to download a CSV containing the application's bulk analysis.

# Data
The raw data used in this project consisted of pre-rated CSV datasets sourced from Kaggle, composed of text customer reviews with ratings provided by the users themselves. These ratings being personally chosen by the users is both a pro and a con; on one hand, the ratings can be seen as highly reflective of the customers' genuine feelings, as they directly express their personal experiences. However, this also introduces an impressionistic quality to the data – customers' ratings are influenced by subjective interpretations, which may not always align with a more objective sentiment measure. Thus, while the data is valuable in capturing authentic user sentiments, the personal individuality when it comes to portraying a 5-point rating system can affect the model's ability to generalize sentiment analysis across varied contexts. To mitigate the effects of this subjectivity in the ratings, future iterations of the project could use expert-reviewed datasets to ensure ratings are less abstract.

Although the system can analyze new inputs, it does not support the immediate retraining of the model with said inputs. Instead, one must add data to and rerun "train_model.py"; through this, an updated model can be output and thus used to replace the one within the web application. Implementation of training via new inputs is possible, but was out of scope for this particular deployment of Mood Sleuth.

# Validation
Mood Sleuth achieved an overall accuracy score of 56%. While this may seem low at first glance, it’s important to consider the factors that influenced this result. Foremost, the data available to train the model could be optimized further in the future, as subjective understandings of rating systems materialize alongside user-generated ratings, making it more challenging for the model to learn a truly objective sentiment pattern. Additionally, the accuracy score is affected by the bot using a 5-point scale instead of a simple binary one – as a result, its predictions being even slightly off lowers the overall accuracy score without clearly illustrating the nuance of the bot's analysis. In practice, the bot often predicts sentiments very close to the correct score, as demonstrated in this table:

| Error Margin    | Percentage of Predictions |
| -------- | ------- |
| 0 points (exact match) | 55.8% |
| 1 point | 35.3% |
| 2 points | 6.8% |
| 3 points | 1.4% |
| 4 points | 0.6% |

This indicates that while the exact match rate is 56%, the model is off by at most 1 point 91% of the time, which is a promising result for real-world use cases.
