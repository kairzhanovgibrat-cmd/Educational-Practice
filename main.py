import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def main():
    print("Loading data...")
    link = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    tablica = pd.read_table(link, header=None, names=['label', 'message'])
    
    tablica['label_num'] = tablica['label'].map({'ham': 0, 'spam': 1})
    
    tekst = tablica['message']
    metki = tablica['label_num']
    
    xTrain, xTest, yTrain, yTest = train_test_split(tekst, metki, test_size=0.2, random_state=42)
    
    vect = CountVectorizer()
    xTrainDtm = vect.fit_transform(xTrain)
    xTestDtm = vect.transform(xTest)
    
    model = MultinomialNB()
    model.fit(xTrainDtm, yTrain)
    
    print("Model ready.\n")
    
    yPred = model.predict(xTestDtm)
    acc = accuracy_score(yTest, yPred)
    print("Accuracy is: " + str(round(acc * 100, 2)) + "%\n")
    
    print("--- AUTOMATIC TEST ---")
    massiv = [
        "WINNER!! You have been selected to receive a $1000 prize. Call now!",
        "Hey bro, are we still meeting for lunch today?",
        "URGENT: Your bank account has been locked. Click here to verify."
    ]
    
    testDtm = vect.transform(massiv)
    preds = model.predict(testDtm)
    
    for i in range(len(massiv)):
        if preds[i] == 1:
            print("[SPAM] " + massiv[i])
        else:
            print("[NOT SPAM] " + massiv[i])
            
    print("\n--- MANUAL TEST ---")
    print("Type 'exit' to stop.")
    
    while True:
        msg = input("\nEnter message: ")
        
        if msg.strip().lower() == 'exit':
            print("Stopping...")
            break
            
        userDtm = vect.transform([msg])
        if model.predict(userDtm)[0] == 1:
            print("=> Result: [SPAM]")
        else:
            print("=> Result: [NOT SPAM]")

if __name__ == "__main__":
    main()
