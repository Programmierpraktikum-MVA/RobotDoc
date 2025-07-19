import os
from io import BytesIO

from flask import jsonify, request, send_file

from util.db_model import db, Patients, Accounts, Image, ChatMessage
from util.exceptions import OccupiedUsernameError, InvalidUsernameError, InvalidPasswordError
from modules.llava_inference import image_captioning_with_robodoc
from modules.subgraphExtractor import processMessage, processWithoutKG, symptomNER

def get_patient(patient_id):
    patient = db.session.get(Patients, patient_id)
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    return jsonify({
        'id': patient.id,
        'name': patient.name,
        'age': patient.age,
        'weight': patient.weight,
        'sex': patient.sex,
        'symptoms': patient.symptoms,
        'user_id': patient.user_id
    }), 200

def get_patient_amount():
    count = db.session.query(Patients).count()
    return jsonify({'total_patients': count}), 200

def get_all_patients():
    patients = Patients.query.all()
    result = [{
        'id': p.id,
        'name': p.name,
        'age': p.age,
        'weight': p.weight,
        'sex': p.sex,
        'symptoms': p.symptoms,
        'user_id': p.user_id
    } for p in patients]
    return jsonify(result), 200

def update_patient_by_id(patient_id, data):
    patient = db.session.get(Patients, patient_id)
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    try:
        required_fields = ['name', 'age', 'weight', 'sex', 'symptoms']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        patient.name = str(data['name']).strip()
        patient.age = int(data['age'])
        patient.weight = float(data['weight'])
        patient.sex = data['sex']
        patient.symptoms = data['symptoms']

        db.session.commit()
        return jsonify({'message': 'Patient updated successfully'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def create_patient(data, user_id):
    name = data.get("name")
    age = int(data.get("age"))
    weight = float(data.get("weight"))
    sex = data.get("sex")
    symptoms = data.get("symptoms")

    if not name or not sex or not symptoms:
        return jsonify({"error": "Missing required fields"}), 400

    new_patient = Patients(
        name=name,
        age=age,
        weight=weight,
        sex=sex,
        symptoms=symptoms,
        user_id=user_id
    )
    db.session.add(new_patient)
    db.session.commit()

    return jsonify({"message": "Patient created", "id": new_patient.id}), 201

def delete_patient_by_id(patient_id):
    patient = Patients.query.get(patient_id)
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    db.session.delete(patient)
    db.session.commit()
    return jsonify({'message': 'Patient deleted successfully'}), 200

def get_image_urls_for_patient(patient_id):
    images = Image.query.filter_by(patient_id=patient_id).all()
    image_urls = [f'/api/image/{img.id}' for img in images]
    return jsonify([{'url': url} for url in image_urls])

def get_image_blob(image_id):
    image = db.session.get(Image, image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    return send_file(
        BytesIO(image.file),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name=f'image_{image_id}.jpg'
    )

def upload_image_for_patient(patient_id):
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_data = file.read()
    if not image_data:
        return jsonify({'error': 'Empty image'}), 400

    new_image = Image(file=image_data, patient_id=patient_id)
    db.session.add(new_image)
    db.session.commit()
    return ('', 204)

def delete_image_by_id(image_id):
    image = db.session.get(Image, image_id)
    if not image:
        return jsonify({'error': 'Image not found'}), 404

    db.session.delete(image)
    db.session.commit()
    return ('', 204)

def save_chat_message(patient_id, sender, message):
    chat = ChatMessage(patient_id=patient_id, sender=sender, message=message)
    db.session.add(chat)
    db.session.commit()
    return ('', 204)

def respond_to_message(patient_id, data):
    try:
        patient = db.session.get(Patients, patient_id)
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404

        message = data.get('message', '').strip()
        update_symptoms = data.get('updateSymptoms', False)
        use_kg = data.get('useKG', True)

        if message:
            save_chat_message(patient_id, sender='user', message=message)

        if update_symptoms and message:
            new_symptoms = symptomNER(message)
            current_symptoms = patient.symptoms or []

            unique_symptoms = [
                s for s in new_symptoms if not any(s in cs for cs in current_symptoms)
            ]

            if unique_symptoms:
                patient.symptoms = current_symptoms + unique_symptoms
                db.session.commit()
                return jsonify({
                    "reply": unique_symptoms,
                    "type": "symptoms"
                }), 200

        patient_info = patient.to_dict()
        if not use_kg:
            reply = processWithoutKG(patient_id, patient_info, message)
        else:
            reply = processMessage(patient_id, patient_info, message)

        save_chat_message(patient_id, sender='robotdoc', message=reply)

        return jsonify({
            "reply": reply,
            "type": "message"
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "reply": f"An error occurred: {str(e)}",
            "type": "error"
        }), 500

def process_uploaded_image_with_llava(patient_id, image_file, message_text):
    try:
        patient = db.session.get(Patients, patient_id)
        if not patient:
            return jsonify({'error': 'Patient not found'}), 404

        if image_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        upload_folder = os.path.join('/app', 'modules', 'img')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, image_file.filename)
        image_file.save(file_path)

        image_file.seek(0)
        image_data = image_file.read()
        if not image_data:
            return jsonify({'error': 'Empty image file'}), 400

        new_image = Image(file=image_data, patient_id=patient_id)
        db.session.add(new_image)
        db.session.commit()

        caption = image_captioning_with_robodoc(image_file.filename)
        patient_info = patient.to_dict()

        save_chat_message(patient_id, sender='user', message=message_text)
        
        reply = processMessage(patient_id, patient_info, message_text, imgCaptioning=caption)

        save_chat_message(patient_id, sender='robotdoc', message=reply)

        return jsonify({'reply': reply, 'type': 'message'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'reply': str(e), 'type': 'error'}), 500

def get_chat_history_for_patient(patient_id):
    messages = (
        db.session.query(ChatMessage)
        .filter_by(patient_id=patient_id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )

    history = []
    for msg in messages:
        role = 'user' if msg.sender.lower() == 'user' else 'robotdoc'
        history.append({'role': role, 'content': msg.message})

    return history

