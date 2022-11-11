from flask import Flask, request, render_template

app=Flask(__name__)
import pickle
count=pickle.load(open("Models/count_Vectorizer.pkl","rb"))
model=pickle.load(open("Models/Movies_Review_Classification.pkl","rb"))

def predict(text):
    sen=count.transform([text]).toarray()
    res=model.predict(sen)[0]
    return res

@app.route("/",methods=["GET","POST"])
def home():
    status=0
    if request.method == 'POST':
        text=request.form.get("message")
        res=predict(text)
        if res==0:
            status=2
        else:
            status=1    
        return render_template("index.html",status=status,text=text)
    return render_template("index.html")
if __name__ =="__main__":
    app.debug=True
    app.run()