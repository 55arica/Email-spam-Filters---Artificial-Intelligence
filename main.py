import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('spam_ham_dataset.csv')

df.dropna()


df['text'] = df['text'].str.lower()   # Convert text to lowercase

# --------------------------------------------------------------------------------------------------------

# Optional: Remove special characters and numbers
df['text'] = df['text'].str.replace(r'[^a-z\s]', '', regex=True)

vectorizer = TfidfVectorizer(stop_words = 'english', max_features = 5000)

# -------------------------------------------------------------------------------------------------------------


x = vectorizer.fit_transform(df['text'])


y = df['label_num']


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)


model = MultinomialNB()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
print(f'accuracy: {accuracy}')

clssfn_report = classification_report(y_test, predictions)

print(f'classification report: {clssfn_report}')





# ---------------------------------------------------------------------------------------------------------------------------


# for user input testing

email = "send me send me your bank info"

email_processed = vectorizer.transform([email.lower()])

prediction = model.predict(email_processed)

result = 'Spam' if prediction[0] ==1 else 'Ham'
print(f'the email is classified as: {result}')
