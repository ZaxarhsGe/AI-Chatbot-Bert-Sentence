# -*- coding: utf-8 -*-
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def encode_sentence(sentence):
    try:
        return model.encode(sentence).reshape(1, -1)
    except Exception as e:
        print(f"Error in encode_sentence: {e}")
        return np.zeros((1, 768))


questions_answers = [
    
    {"question":"Αναζητώ μαξιλάρια χτιστού καναπέ / πάγκου", "answer": "Μπορείτε να βρείτε διάφορες φωτογραφίες από δουλειές μας εδω: https://maxilaria.gr/"},
    {"question":"I’m looking for cushions for built-in sofas/benches.", "answer": "You can find various photos of our work here: https://maxilaria.gr/"},
    {"question":"Μπορείτε να αναλάβετε την αλλαγή ταπετσαρίας σε έπιπλα καναπέ - πολυθρόνα;", "answer": "Βεβαίως μπορείτε να μας στείλετε φωτογραφίες του καναπέ ή των επίπλων σας, τις διαστάσεις τους και την περιοχή - όροφο που βρίσκονται στο email maxilaria.gr@gmail.com και θα σας απαντήσουμε λεπτομερώς με τη προσφορά μας."},
    {"question":"Can you undertake the change of upholstery on furniture sofa-armchair?", "answer": "Of course you can send us photos of your sofa or furniture, their dimensions and the area - floor where they are located to the email maxilaria.gr@gmail.com and we will answer you in detail with our offer."},
    {"question":"Κάνετε επισκευή επίπλων bamboo/ μπαμπού;", "answer": "Βεβαίως! Στέλνετε στο maxilaria.gr@gmail.com, φωτογραφίες των επίπλων, αναγράφεται τα τεμάχια και την περιοχή που αυτά βρίσκονται και σας απαντούμε με τη προσφορά μας."},
    {"question":"Do you repair bamboo furniture?", "answer": "Of course! Send photos of the furniture to maxilaria.gr@gmail.com, indicate the pieces and the area where they are located and we will reply with our offer."},
    {"question":"Αναλαμβάνεται την επισκευή σε έπιπλα Rattan / Wicker ;", "answer": "Αναλαμβάνουμε την επισκευή σε έπιπλα μόνο σε φυσικό Rattan. Δυστυχώς δεν αναλαμβάνουμε την επισκευή σε έπιπλα από πλαστικοποιημένο Rattan / Wicker."},
    {"question":"Do you undertake the repair of Rattan / Wicker furniture?", "answer": "We undertake repairs on natural Rattan furniture only. Unfortunately we do not undertake repairs on laminated Rattan / Wicker furniture."},
    {"question":"Που μπορώ να βρω μαξιλάρια παλέτας;", "answer": "Μπορείτε να βρείτε διάφορες κατασκευές μας εδώ : https://maxilaria.gr/paletes/"},
    {"question":"Where can I find pallet pillows?", "answer": "You can find our various constructions here: https://maxilaria.gr/paletes/"},
    {"question":"Έχετε ετοιμοπαράδοτα μαξιλάρια ξαπλώστρας;", "answer": "Όχι, Τα μαξιλάρια ξαπλώστρας είναι διαθέσιμα κατόπιν παραγγελίας. Θα χρειαστεί να μας στείλετε στο maxilaria.gr@gmail.com, τις διαστάσεις που έχει η ξαπλώστρα σας, Το συνολικό μήκος, το σημείο που σπάει η πλάτη, το βάθος και το επιθυμητό πάχος για να σας απαντήσουμε με τη προσφορά μας."},
    {"question":"Do you have sunbed cushions ready for delivery?", "answer": "No, sunbed cushions are available upon request. You will need to send us to maxilaria.gr@gmail.com the dimensions of your sunbed, the total length, the point where the back breaks, the depth and the desired thickness in order to respond with our offer."},
    {"question":"Που μπορώ να βρω μαξιλάρια για επαγγελματική χρήση;", "answer": "Μπορείτε να βρείτε τα προϊόντα μας για επαγγελματική χρήση εδώ: https://maxilaria.gr/maxilaria-epaggelmatikis-xrisis/"},
    {"question":"Where can I find pillows for professional use?", "answer": "You can find our products for professional use here: https://maxilaria.gr/maxilaria-epaggelmatikis-xrisis/"},
    {"question":"Μπορείτε να κατασκευάσετε πουφ στις διαστάσεις που επιθυμώ/ θέλω;", "answer": "Βεβαίως. Στείλτε μας τις διαστάσεις και πιθανόν ενδεικτικές φωτογραφίες από το πουφ που θέλετε να κατασκευαστεί και θα σας στείλουμε τη προσφορά μας. Επίσης μπορείτε να δείτε κάποιες κατασκευές μας εδω: https://maxilaria.gr/maxilaria-diakosmitika-dapedon-pouf/"},
    {"question":"Can you make poufs in the dimensions I want/want?", "answer": "Of course. Send us the dimensions and possibly indicative photos of the pouf you want manufactured and we will send you our offer. You can also see some of our constructions here: https://maxilaria.gr/maxilaria-diakosmitika-dapedon-pouf/"},
    {"question":"Υπάρχουν πουφ μαξιλάρες ξαπλώστρας;", "answer": "Βεβαίως. Υπάρχουν 2 κατασκευές για μαξιλάρες πουφ ξαπλώστρας. Η Gem και η Εva. Μπορείτε να τις βρείτε εδώ: https://maxilaria.gr/maxilaria-epaggelmatikis-xrisis/"},
    {"question":"Are there pouf sunbed cushions?", "answer": "Of course. There are 2 designs for lounger pouf cushions. Gem and Eva. You can find them here: https://maxilaria.gr/maxilaria-epaggelmatikis-xrisis/"},
    {"question":"Υπάρχει κάτι διαθέσιμο σε στυλ πουφ πολυθρόνας;", "answer": "Εννοείται! Η Sissy, η Angie, η Donna, η Thone και η Pyramis μπορούν να ανταποκριθούν σε κάθε σας ανάγκη. Τις βρίσκετε πηγαίνοντας στη κατηγορία πουφ."},
    {"question":"Is there anything available in the ottoman style?", "answer": "Of course! Sissy, Angie, Donna, Thone and Pyramis can meet your every need. He finds them by going to the pouf category."},
    {"question":"Κάνετε κατασκευή μαξιλαριών Daybed;", "answer": "Βεβαίως. Μπορείτε να μας στείλετε τις διαστάσεις μήκος - βάθος - πάχος και κάποιες φωτογραφίες από το Daybed που θέλετε να κατασκευάσουμε και θα σας απαντήσουμε με τη προσφορά μας."},
    {"question":"Do you manufacture Daybed cushions?", "answer": "Of course. You can send us the dimensions length-depth-thickness and some photos of the Daybed you want us to make and we will answer you with our offer."},
    {"question":"Κατασκευάζεται μαξιλάρια ύπνου;", "answer": "Δυστυχώς όχι."},
    {"question":"Do you manufacture sleeping pads?", "answer": "Unfortunately no."},
    {"question":"Φτιάχνεται τραπεζομάντηλα;", "answer": "Βεβαίως! Μπορείτε να μας στείλετε τις επιθυμητές διαστάσεις στο maxilaria.gr@gmail.com"},
    {"question":"Do you manufacture tablecloths?", "answer": "Of course! You can send us the desired dimensions to maxilaria.gr@gmail.com"},
    {"question":"Που μπορώ να βρώ ύφασμα για τραπεζομάντηλα;", "answer": "Μπορείτε να βρείτε κάποια από τα υφάσματά μας εδω: https://maxilaria.gr/trapezomantila/ ή να μας καλέσετε στο 2108975114 για να κλείσετε το ραντεβού σας για να τα δείτε από κοντά."},
    {"question":"Where can I find tablecloth fabric?", "answer": "You can find some of our fabrics here: https://maxilaria.gr/trapezomantila/ or call us at 2108975114 to make an appointment to see them in person."},
    {"question":"Φτιάχνετε μαξιλάρια για πολυθρόνα bamboo ή για έπιπλα εξωτερικού χώρου;", "answer": "Βεβαίως. Μπορείτε να δείτε κάποιες κατασκευές μας εδώ: https://maxilaria.gr/maxilaria-kanape-polythronas/"},
    {"question":"Do you make cushions for bamboo armchairs or for outdoor furniture?", "answer": "Of course. You can see some of our constructions here: https://maxilaria.gr/maxilaria-kanape-polythronas/"},
    {"question":"Κατασκευάζεται καλύμματα για καρέκλες;", "answer": "Βεβαίως! Μπορείτε να έχετε μια εικόνα από διάφορες παρόμοιες δουλειές μας εδω: https://maxilaria.gr/maxilaria-kareklas/"},
    {"question":"Do you manufacture chair covers?", "answer": "Of course! You can have an image of our various similar jobs here: https://maxilaria.gr/maxilaria-kareklas/"},
    {"question":"Έχετε καλαμωτές;", "answer": "Βεβαίως: Μπορείτε να δείτε όλες τις διαθέσιμες πληροφορίες εδω: https://maxilaria.gr/kalamotes/"},
    {"question":"Do you have strawmen?", "answer": "Of course: You can see all the available information here: https://maxilaria.gr/kalamotes/"},
    {"question":"Πως μπορω να παραγγείλω;", "answer": "Για να μπορέσει να πραγματοποιηθεί μια παραγγελία θα πρέπει να μας στείλετε email στο maxilaria.gr@gmail.com, ένα μήνυμα σχετικά με αυτό που θέλετε να κατασκευάσετε και τις επιθυμητές διαστάσεις, είτε μπορείτε να καλέσετε στο 2108975114 για να κλείσετε το ραντεβού και να τα δούμε όλα μαζί από κοντά. Θα χαρούμε πολύ να σας εξυπηρετήσουμε."},
    {"question":"How can I order?", "answer": "To be able to place an order you should send us an email at maxilaria.gr@gmail.com, a message about what you want to build and the desired dimensions, or you can call 2108975114 to make an appointment and let's see it all together up close. We will be very happy to serve you."},
    {"question":"Έχετε ή κατασκευάζετε μαξιλάρια για καρέκλες φερ φορζε;", "answer": "Οι καρέκλες φερ φορζε έχουν μεταξύ τους μικροδιαφορές που δεν μας επιτρέπουν να έχουμε ετοιμοπαράδοτα μαξιλάρια. Μπορείτε να μας στείλετε μια φωτογραφία της καρέκλας που έχετε στο maxilaria.gr@gmail.com και τις διαστάσεις ώστε να σας ενημερώσουμε με τη προσφορά μας."},
    {"question":"Do you have or manufacture ferforze chair cushions?", "answer": "Ferforge chairs have minor differences between them that do not allow us to have ready-made cushions. You can send us a photo of the chair you have at maxilaria.gr@gmail.com and the dimensions so that we can inform you of our offer."},
    {"question":"Κατασκευάζετε καλύμματα που προστατεύουν τα έπιπλα εξωτερικού χώρου;", "answer": "Βεβαίως! Μπορείτε αρχικά να δείτε κάποιες φωτογραφίες από δουλειές μας εδω: https://maxilaria.gr/kalymmata-prostasias/ και έπειτα να μας στείλετε μια φωτογραφία των δικών σας επίπλων και τις διαστάσεις τους στο maxilaria.gr@gmail.com ώστε να σας ενημερώσουμε με τη προσφορά μας."},
    {"question":"Do you manufacture covers that protect outdoor furniture?", "answer": "Of course! You can first see some photos of our work here: https://maxilaria.gr/kalymmata-prostasias/ and then send us a photo of your furniture and its dimensions to maxilaria.gr@gmail.com so we can inform you with our offer."},
    {"question":"Αναλαμβάνετε την δημιουργία / κατασκευή κουρτινών;", "answer": "Βεβαίως! Θα χρειαστεί να μας στείλετε στο maxilaria.gr@gmail.com τα στοιχεία σας ώστε να επικοινωνήσει μαζί σας ο συνεργάτης μας για να πραγματοποιήσει τις μετρήσεις και έπειτα κλείνουμε ραντεβού στο κατάστημά μας ώστε να επιλέξετε τα κατάλληλα υφάσματα για εσάς και να προχωρήσουμε στη παραγγελία σας. Η παραγγελία ολοκληρώνεται με την τοποθέτηση των κουρτινών."},
    {"question":"Do you undertake the creation / manufacture of curtains?", "answer": "Of course! You will need to send us your details at maxilaria.gr@gmail.com so that our partner can contact you to take the measurements and then we make an appointment at our store so that you can choose the right fabrics for you and proceed with your order. The order is completed with the installation of the curtains."}
]


questions = [qa["question"] for qa in questions_answers]
answers = [qa["answer"] for qa in questions_answers]
encoded_questions = np.array([encode_sentence(question) for question in questions]).squeeze()

def chatbot_response(user_input):
    try:
        user_encoded = encode_sentence(user_input).reshape(1, -1)
        similarities = cosine_similarity(user_encoded, encoded_questions).flatten()
        print(f"Similarities: {similarities}")

        
        threshold = 0.6
        most_similar_index = similarities.argmax()
        most_similar_score = similarities[most_similar_index]

        if most_similar_score > threshold:
            return answers[most_similar_index]
        else:
            return "I didn't quite understand your question. Did you mean: '{}'?".format(questions[most_similar_index])
    except Exception as e:
        print(f"Error in chatbot_response: {e}")
        return "Sorry, I couldn't process your request."


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message")
        response = chatbot_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"response": "Sorry, there was an error processing your request."})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
