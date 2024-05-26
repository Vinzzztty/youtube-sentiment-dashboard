import googleapiclient.discovery
from config import get_secret_key
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import re

# Ensure you have the necessary NLTK data files
nltk.download("stopwords")


class YoutubeCrawler:
    def __init__(self):
        self.api_service_name = "youtube"
        self.api_version = "v3"

    # Get Comment
    def crawl_comments(self, video_id, req_result):

        youtube = googleapiclient.discovery.build(
            self.api_service_name,
            self.api_version,
            developerKey=get_secret_key(),
        )

        max_result = int(req_result)

        results = []

        nextPageToken = None

        while len(results) < max_result:
            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=nextPageToken,
            )
            response = req.execute()

            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comment_text = re.sub(
                    r"<.*?>", "", comment["textDisplay"]
                )  # Remove HTML tags
                author_without_at = re.sub(
                    r"@", "", comment["authorDisplayName"]
                )  # Remove '@' symbol

                if (
                    len(comment_text) >= 12
                ):  # Only add comments with at least 20 characters
                    results.append(
                        {
                            "author": author_without_at,
                            "like_count": comment["likeCount"],
                            "comment": comment_text,
                            "authorImage": comment["authorProfileImageUrl"],
                        }
                    )

                if (
                    len(results) >= max_result
                ):  # Break if we have collected enough comments
                    break

            nextPageToken = response.get("nextPageToken")
            if not nextPageToken:  # Break the loop if no more pages are available
                break

        result_processor = ResultProcessor(results)
        html_output = result_processor.normalize_alay()

        return html_output


class ResultProcessor:
    def __init__(self, result):
        self.result = result

    def raw_result(self):
        df = pd.DataFrame(self.result)

        df.to_csv("./results/raw_results.csv")

    def normalize_alay(self):
        alay_dict = pd.read_csv(
            "https://raw.githubusercontent.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/master/new_kamusalay.csv",
            names=["original", "replacement"],
            encoding="latin-1",
        )
        alay_dict_map = dict(zip(alay_dict["original"], alay_dict["replacement"]))

        for item in self.result:
            item["comment_normalized"] = " ".join(
                alay_dict_map.get(word, word) for word in item["comment"].split()
            )

        return self.process_to_html()

    def process_to_html(self):

        # Convert result to DataFrame
        new_df = pd.DataFrame(self.result)

        # Rename columns and format HTML for the UserImage
        if "authorImage" in new_df.columns:
            new_df["UserImage"] = new_df["authorImage"].apply(
                lambda x: f'<img src="{x}" width="50" height="50">'
            )
        else:
            new_df["UserImage"] = (
                f'<img src="default_image_url_here" width="50" height="50">'
            )

        df_process = new_df[["UserImage", "author", "comment_normalized", "like_count"]]
        print(len(df_process["author"]))

        # Rename columns for HTML presentation
        df_process.rename(
            columns={
                "UserImage": "User Images",
                "author": "Username",
                "comment_normalized": "Review",
                "like_count": "Like",
            },
            inplace=True,
        )

        # Wordcloud
        all_reviews_text = " ".join(df_process["Review"])

        wordcloud_generator = WordCloudGenerator()
        wordcloud_generator.generate_wordcloud(all_reviews_text)

        # Use Pandas to_html to convert DataFrame to HTML table
        table_classes = "table table-responsive table-striped text-center"
        html_output = df_process.to_html(
            index=False, escape=False, header=True, classes=table_classes
        )

        return html_output


class WordCloudGenerator:
    def __init__(self, custom_words=None):
        self.custom_words = custom_words if custom_words else []

    # Function to remove stopwords
    def remove_stopwords(self, text, language="indonesian"):
        stop_words = set(stopwords.words(language))
        return " ".join(
            [word for word in text.split() if word.lower() not in stop_words]
        )

    # Function to remove custom words from text
    def remove_custom_words(self, text):
        for word in self.custom_words:
            text = text.replace(word, "")
        return text

    # Function to remove numbers and symbols from text
    def remove_numbers_and_symbols(self, text):
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text

    def preprocess_text(self, text):
        text = self.remove_stopwords(text)
        text = self.remove_custom_words(text)
        text = self.remove_numbers_and_symbols(text)
        return text

    def generate_wordcloud(self, text, save_path="./static/images/wordcloud.png"):
        preprocessed_text = self.preprocess_text(text=text)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            random_state=3,
            max_font_size=110,
            stopwords=STOPWORDS,
        ).generate(preprocessed_text)

        matplotlib.use("Agg")

        # Display the generated WordCloud using matplotlib
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        # Save the generated WordCloud to a file
        plt.savefig(save_path, bbox_inches="tight")

        plt.close()
