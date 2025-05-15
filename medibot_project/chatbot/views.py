from django.shortcuts import render
import joblib
import pandas as pd
from unidecode import unidecode
from rdflib import Graph, Namespace, RDFS

# Load model & data
vectorizer = joblib.load("./chatbot/model/tfidf_vectorizer.pkl")
model_SVM = joblib.load("./chatbot/model/SVM.pkl")
model_RF = joblib.load("./chatbot/model/RF.pkl")
mlb = joblib.load("./chatbot/model/mlb.pkl")
loaded_tfidf = joblib.load("./chatbot/model/tfidf_vectorizer.pkl")
df = pd.read_csv('./chatbot/data/data.csv')

# Load ontology
g = Graph()
g.parse('./chatbot/data/disease_ontology.owl', format='xml')
n = Namespace("http://example.com/health_qa#")

# Nhận diện nhiều intent trong một câu hỏi
def classify_question(question, model_type="SVM"):
    question_tfidf = loaded_tfidf.transform([question])
    
    if model_type == "RF":
        y_pred = model_RF.predict(question_tfidf)
    else:  # Mặc định là SVM
        y_pred = model_SVM.predict(question_tfidf)
    
    detected_intents = mlb.inverse_transform(y_pred)[0]
    
    return detected_intents

# Tìm tên bệnh từ câu hỏi
def extract_disease_name(question, disease_list):
    question_no_accent = unidecode(question.lower())
    best_match = None
    max_length = 0
    
    for disease in disease_list:
        disease_no_accent = unidecode(disease.lower())
        if disease_no_accent in question_no_accent:
            if len(disease_no_accent) > max_length:
                best_match = disease
                max_length = len(disease_no_accent)
    
    return best_match

# Truy vấn ontology với từng intent
def query_ontology(disease, intent):
    intent_map = {
        "Triệu chứng": "Symptom",
        "Điều trị": "Treatment",
        "Nguyên nhân": "Cause"
    }
    owl_type = intent_map.get(intent, "")
    if not owl_type:
        return ["Không rõ loại câu hỏi"]

    query = f"""
    PREFIX health: <{n}>
    PREFIX rdfs: <{RDFS}>

    SELECT ?answer WHERE {{
        ?question a health:Question ;
                  rdfs:label ?label ;
                  health:hasAnswer ?aObj .
        FILTER(CONTAINS(LCASE(?label), "{disease.lower()}") && CONTAINS(LCASE(?label), "{intent.lower()}"))
        ?aObj rdfs:label ?answer .
    }}
    """
    results = g.query(query)
    return [str(row.answer) for row in results] or ["Không có thông tin"]

# View chính của chatbot
def chatbot_view(request):
    if 'chat_history' not in request.session:
        request.session['chat_history'] = []

    chat_history = request.session['chat_history']
    answer = ""
    disease = None
    selected_model = "SVM"
    user_input = None

    if request.method == "POST":
        user_input = request.POST.get("user_input")
        selected_model = request.POST.get("model_type", "SVM")
        disease_list = sorted(df['Name'].unique().tolist())
        
        intents = classify_question(user_input, selected_model)
        disease = extract_disease_name(user_input, disease_list)

        if not disease:
            answer = "Xin lỗi, tôi không nhận diện được tên bệnh trong câu hỏi."
        else:
            all_answers = []
            for intent in intents:
                result = query_ontology(disease, intent)
                formatted = f"<b>{intent}:</b><br>" + "<br>".join(result)
                all_answers.append(formatted)
            answer = "<br><br>".join(all_answers)

        chat_entry = {
            'user_input': user_input,
            'disease': disease,
            'answer': answer,
        }
        chat_history.append(chat_entry)
        request.session['chat_history'] = chat_history  

    context = {
        'chat_history': chat_history,
        'user_input': user_input,
        'disease': disease,
        'answer': answer,
        'selected_model': selected_model,
    }
    return render(request, "chatbot/chat.html", context)