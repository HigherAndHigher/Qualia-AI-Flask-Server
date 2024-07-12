import os
from flask import Flask, request, jsonify
from langchain.llms import OpenAI
from flask_cors import CORS
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import uuid

app = Flask(__name__)
CORS(app)

# Firestore DB初期化
cred = credentials.Certificate('./key.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

#　環境変数
openai_api_key = os.environ.get('OPENAI_API_KEY')

@app.route('/create', methods=['POST'])
def create_ai_friend():
    data = request.json
    friend_id = str(uuid.uuid4())
    ai_friend_data = {
      'id': friend_id,
      'model_name': data.get('model_name'),
      'entry_name': data.get('entry_name'),
      'avatar_url': data.get('avatar_url'),
      'name': data.get('name'),
      'birthday': data.get('birthday'),
      'hometown': data.get('hometown'),
      'gender': data.get('gender'),
      'bloodtype': data.get('blood_type'),
      'family_name': data.get('family_names'),
      'occupation': data.get('occupation'),
      'hobby': data.get('hobbies'),
      'special_skills': data.get('special_skills'),
      'self_introduction': data.get('self_introduction'),
      'first_person_pronoun': data.get('first_person_pronoun'),
      'conversation_style': data.get('conversation_style'),
      'conversation_starters': data.get('conversation_starters'),
      'conversation_endings': data.get('conversation_endings'),
      'intonation_and_volume': data.get('intonation_and_volume'),
      'conversation_pacing': data.get('conversation_pacing'),
      'fluency': data.get('fluency'),
      'dialect': data.get('dialect'),
      'languages_spoken': data.get('languages_spoken'),
      'other_speaking_styles': data.get('other_speaking_styles'),
      'favorite_animal': data.get('favorite_animal'),
      'favorite_book': data.get('favorite_book'),
      'favorite_movie': data.get('favorite_movie'),
      'favorite_singer_or_talent': data.get('favorite_singer_or_talent'),
      'favorite_song': data.get('favorite_song'),
      'favorite_sport': data.get('favorite_sport'),
      'favorite_game': data.get('favorite_game'),
      'favorite_artwork_or_anime': data.get('favorite_artwork_or_anime'),
      'favorite_color': data.get('favorite_color'),
      'favorite_word': data.get('favorite_word'),
      'favorite_season': data.get('favorite_season'),
      'favorite_travel_destination': data.get('favorite_travel_destination'),
      'solo_travel_or_group_travel': data.get('solo_travel_or_group_travel'),
      'favorite_and_disliked_foods_and_drinks': data.get('favorite_and_disliked_foods_and_drinks'),
      'spicy_or_sweet_food_preference': data.get('spicy_or_sweet_food_preference'),
      'cooking_skills_and_signature_dishes': data.get('cooking_skills_and_signature_dishes'),
      'favorite_possessions_or_gadgets': data.get('favorite_possessions_or_gadgets'),
      'collected_items': data.get('collected_items'),
      'most_expensive_purchase': data.get('most_expensive_purchase'),
      'favorite_purchase': data.get('favorite_purchase'),
      'stress_relief_methods': data.get('stress_relief_methods'),
      'definition_of_family': data.get('definition_of_family'),
      'favorite_place_at_home': data.get('favorite_place_at_home'),
      'favorite_event': data.get('favorite_event'),
      'weekend_activities': data.get('weekend_activities'),
      'morning_or_night_person': data.get('morning_or_night_person'),
      'indoor_or_outdoor_person': data.get('indoor_or_outdoor_person'),
      'active_or_laidback_person': data.get('active_or_laidback_person'),
      'optimist_or_pessimist': data.get('optimist_or_pessimist'),
      'planning_or_spontaneous_person': data.get('planning_or_spontaneous_person'),
      'recent_obsession': data.get('recent_obsession'),
      'recent_happiness': data.get('recent_happiness'),
      'recent_sadness': data.get('recent_sadness'),
      'recent_anger': data.get('recent_anger'),
      'recent_emotional_moment': data.get('recent_emotional_moment'),
      'recent_learning': data.get('recent_learning'),
      'recent_regret': data.get('recent_regret'),
      'recent_reflection': data.get('recent_reflection'),
      'recent_gratitude': data.get('recent_gratitude'),
      'recent_appreciation_received': data.get('recent_appreciation_received'),
      'one_word_description': data.get('one_word_description'),
      'animal_representation': data.get('animal_representation'),
      'childhood_personality': data.get('childhood_personality'),
      'dream_job_as_a_child': data.get('dream_job_as_a_child'),
      'moments_of_happiness': data.get('moments_of_happiness'),
      'point_of_pride': data.get('point_of_pride'),
      'habit_to_change': data.get('habit_to_change'),
      'memorable_feedback': data.get('memorable_feedback'),
      'life_changing_event': data.get('life_changing_event'),
      'future_challenges': data.get('future_challenges'),
      'anger_traits': data.get('anger_traits'),
      'sadness_traits': data.get('sadness_traits'),
      'lying_traits': data.get('lying_traits'),
      'struggling_traits': data.get('struggling_traits'),
      'happiness_traits': data.get('happiness_traits'),
      'laughing_style': data.get('laughing_style'),
      'crying_style': data.get('crying_style'),
      'joke_telling_frequency': data.get('joke_telling_frequency'),
      'abstract_expression_preference': data.get('abstract_expression_preference'),
      'personal_motto': data.get('personal_motto'),
      'earliest_memory': data.get('earliest_memory'),
      'childhood_dream': data.get('childhood_dream'),
      'first_love': data.get('first_love'),
      'person_who_understands_you_best': data.get('person_who_understands_you_best'),
      'comforting_place': data.get('comforting_place'),
      'favorite_physical_feature': data.get('favorite_physical_feature'),
      'commonly_misunderstood_aspect': data.get('commonly_misunderstood_aspect'),
      'overcome_difficulty': data.get('overcome_difficulty'),
      'impactful_life_event': data.get('impactful_life_event'),
      'valued_relationship': data.get('valued_relationship'),
      'cherished_traditions': data.get('cherished_traditions'),
      'important_lesson_to_share': data.get('important_lesson_to_share'),
      'message_to_humanity': data.get('message_to_humanity'),
      'desired_superpower': data.get('desired_superpower'),
      'moment_to_relive': data.get('moment_to_relive'),
      'advice_to_past_self': data.get('advice_to_past_self'),
      'biggest_regret': data.get('biggest_regret'),
      'final_words': data.get('final_words'),
      'desired_legacy': data.get('desired_legacy'),
      'ideal_reincarnation': data.get('ideal_reincarnation')
    }

    db.collection('ai_friends').document(friend_id).set(ai_friend_data)
    
    return jsonify({'friend_id': friend_id})

@app.route('/delete', methods=['POST'])
def delete_ai_friend():
    friend_id = request.json.get('friend_id')
    
    doc_ref = db.collection('ai_friends').document(friend_id)
    if doc_ref.get().exists:
        doc_ref.delete()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'not found'}), 404

@app.route('/user_info', methods=['POST'])
def user_info():
    data = request.json
    
    user_data = {
        'name': data['user_name'],
        'relationship': data['relationship'],
        'visit_purpose': data['visit_purpose']
    }
    
    friend_id = data['friend_id']
    db.collection('ai_friends').document(friend_id).collection('user_sessions').document('current_user').set(user_data)
    
    return jsonify({'status': 'success'})

@app.route('/chat', methods=['POST'])
def chat_with_ai_friend():
    data = request.json
    friend_id = data.get('friend_id')
    user_message = data.get('message')
    
    doc_ref = db.collection('ai_friends').document(friend_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return jsonify({'status': 'not found'}), 404
    
    ai_friend_data = doc.to_dict()
    
    user_doc_ref = doc_ref.collection('user_sessions').document('current_user')
    user_doc = user_doc_ref.get()
    
    if not user_doc.exists:
        return jsonify({'status': 'user not found'}), 404
    
    user_data = user_doc.to_dict()

    # チャット履歴の取得
    chat_history_ref = doc_ref.collection('chat_history').order_by('timestamp')
    chat_history_docs = chat_history_ref.stream()
    chat_history = [{'role': doc.to_dict()['role'], 'message': doc.to_dict()['message']} for doc in chat_history_docs]

    # AIが友人の反応を生成
    response = generate_response(ai_friend_data, user_data, user_message, chat_history)

    # ユーザーメッセージとボットの応答をチャット履歴に保存
    chat_ref = doc_ref.collection('chat_history').document()
    chat_ref.set({
        'role': 'user',
        'message': user_message,
        'timestamp': firestore.SERVER_TIMESTAMP
    })
    chat_ref = doc_ref.collection('chat_history').document()
    chat_ref.set({
        'role': 'bot',
        'message': response,
        'timestamp': firestore.SERVER_TIMESTAMP
    })
    
    return jsonify({'response': response})

@app.route('/delete_chat_history', methods=['POST'])
def delete_chat_history():
    friend_id = request.json.get('friend_id')
    
    doc_ref = db.collection('ai_friends').document(friend_id)
    chat_history_ref = doc_ref.collection('chat_history')
    chat_history_docs = chat_history_ref.stream()
    
    for doc in chat_history_docs:
        doc.reference.delete()
    
    return jsonify({'status': 'chat history deleted'})

def generate_response(ai_friend_data, user_data, user_message, chat_history):
    openai_api = OpenAI(api_key = openai_api_key)
    
    chat_history_str = '\n'.join([f"{entry['role']}: {entry['message']}" for entry in chat_history])
    prompt = f"あなたは{ai_friend_data['name']}です。\n"
    prompt += f"あなたの情報: {ai_friend_data}\n"
    prompt += f"ユーザーの名前: {user_data['name']}\n"
    prompt += f"ユーザーとの関係: {user_data['relationship']}\n"
    prompt += f"訪問目的: {user_data['visit_purpose']}\n"
    prompt += f"チャット履歴:\n{chat_history_str}\n"
    prompt += f"ユーザーのメッセージ: {user_message}\n"
    prompt += "AIの情報とチャット履歴に基づいて自然に応答してください。自分がAIであることを言わないでください。\n"
    prompt += "自然に応答してください。応答に「AIのメッセージ」や「AIの応答」、AI自分の名前などの前置きを入れないでください。応答にユーザーのメッセージも入れないでください。また、応答に自身の名前や「:」を入れないでください。\n"
    prompt += "自然に応答してください。応答に「AIのメッセージ」や「AIの応答」、自身の名前、「bot:」や「:」などの前置きを入れないでください。\n"

    response = openai_api(prompt, max_tokens=150)
    # レスポンスに不要な接頭辞が含まれないようにする。
    if(response.startswith('AIのメッセージ') or response.startswith('AIの応答') or response.startswith('AI自分の名前') or response.startswith(':')):
        response = response[1:]
    
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)