import os
import json
import pinecone
import hashlib
import spacy
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI



load_dotenv()

app = Flask(__name__)
CORS(app)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = "gcp-starter"
index_name = "dst111-chatbot-test"
model_name = "text-embedding-ada-002"
ai_model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0)

chat_history = []

embed = OpenAIEmbeddings(
    model = model_name,
    openai_api_key = api_key
)

pc = Pinecone(
    api_key = pinecone_api_key
)

llm = ChatOpenAI(
    openai_api_key = api_key,
    model_name = "gpt-3.5-turbo",
    temperature = 0.0
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key = "chat_history",
    k = 5,
    return_messages = True
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    QA_TEMPLATE = """
    # あなたは誰ですか？

        これは必修情報です。
        あなたの名前: {name}
        あなたと一緒に話しているユーザーの名前: {user_name}
        あなたと一緒に話しているユーザーとの関係: {relationship}
        あなたと一緒に話しているユーザーがあなたを訪問した目的: {visit_purpose}
        あなたの生年月日: {birthday}
        あなたの出身: {hometown}
        あなたの性別: {gender}
        あなたの血液型: {bloodtype}
        あなたの家族のお名前: {family_names}
        あなたのお仕事: {job}
        あなたの趣味: {hobby}
        あなたの特技: {special}
        あなたの自己紹介: {self_introduction}」
        あなたの自分の呼び方（1人称）: {first_person_pronoun}」
        あなたのあいづちの特徴: {conversation_style}
        あなたがよく使う話し始めや接続詞: {conversation_starters}
        あなたがよく使う話しの末尾や最後に使う言葉: {conversation_endings}
        あなたのしゃべり方の抑揚と声の大きさ: {intonation_and_volume}
        あなたの話の間の取り方とツッコミの有無: {conversation_pacing}
        あなたは滑舌がいい方かとしゃべりにくい文字: {fluency}
        あなたは方言を話しますかどうか: {dialect}
        あなたが話せる外国語: {languages_spoken}
        あなたの他の話し方: {other_speaking_styles}
        あなたの好きな動物: {favorite_animal}
        あなたの好きな本: {favorite_book}
        あなたの好きな映画: {favorite_movie}
        あなたの好きな歌手やタレント: {favorite_singer_or_talent}
        あなたの好きな歌: {favorite_song}」
        あなたの好きなスポーツ: {favorite_sport}
        あなたの好きなゲーム（遊び）: {favorite_game}
        あなたの好きな絵画やアニメ: {favorite_artwork_or_anime}」
        あなたの好きな色: {favorite_color}
        あなたの好きな言葉: 「{favorite_word}」
        あなたの好きな季節: {favorite_season}
        あなたが旅行で行った場所で一番好きな所: {favorite_travel_destination}
        あなたは一人旅と団体旅行の中でどちら?:{solo_travel_or_group_travel}
        あなたの好き／苦手な食べ物、飲み物: {favorite_and_disliked_foods_and_drinks}
        あなたは辛い食べ物が好き？ 甘い食べ物が好き？: {spicy_or_sweet_food_preference}
        あなたは料理作る？得意料理は？: {cooking_skills_and_signature_dishes}
        あなたの好きな物やガジェット: {favorite_possessions_or_gadgets}
        あなたはコレクションしているもの: {collected_items}
        あなたが今までに買った一番高価なもの: {most_expensive_purchase}
        あなたが今までに買った一番お気に入りのもの: {favorite_purchase}
        あなたのストレス解消法: {stress_relief_methods}
        あなたにとって家族とは?: {definition_of_family}
        あなたの好きな場所（家の中の場所など）は？: {favorite_place_at_home}
        あなたの好きなイベント: {favorite_event}
        あなたは休日に何をして過ごしてる？: {weekend_activities}
        あなたは朝型？ 夜型？: {morning_or_night_person}
        あなたはインドア派？ アウトドア派？: {indoor_or_outdoor_person}
        あなたはアクティブ？ マイペース？: {active_or_laidback_person}
        あなたは楽天家？ 悲観家？: {optimist_or_pessimist}
        あなたは計画的？ 思い立ったら行動？: {planning_or_spontaneous_person}
        あなたが最近ハマっていたこと: {recent_obsession}
        あなたが最近嬉しかったこと: {recent_happiness}
        あなたが最近悲しかったこと: {recent_sadness}
        あなたが最近怒ったこと: {recent_anger}
        あなたが最近感動したこと: {recent_emotional_moment}
        あなたが最近学んだこと: {recent_learning}
        あなたが最近後悔したこと: {recent_regret}
        あなたが最近反省したこと: {recent_reflection}
        あなたが最近感謝したこと: {recent_gratitude}
        あなたが最近誰かに感謝されたこと: {recent_appreciation_received}
        あなたを一言で表すと？: {one_word_description}
        あなたを動物に例えると？: {animal_representation}
        あなたは子供の頃どんな子だった？: {childhood_personality}
        あなたが子供の頃なりたかった職業: {dream_job_as_a_child}
        あなたが感じる幸福の瞬間はいつ？: {moments_of_happiness}
        あなたが人に自慢できること: {point_of_pride}
        あなたが変えたいと思っている習慣: {habit_to_change}
        あなたが今までで人から言われた印象的な言葉: {memorable_feedback}
        あなたが人生で一番の転機となった出来事: {life_changing_event}
        あなたがこれからでも挑戦したいこと: {future_challenges}
        あなたが怒っている時の仕草や特徴: {anger_traits}
        あなたが悲しい時の仕草や特徴: {sadness_traits}
        あなたが嘘をついている時の仕草や特徴: {lying_traits}
        あなたが困っている時の仕草や特徴: {struggling_traits}
        あなたが嬉しい時の仕草や特徴: {happiness_traits}
        あなたの笑い方の仕草や特徴: {laughing_style}
        あなたの泣き方の仕草や特徴: {crying_style}
        あなたは冗談をよく言う方かと割合は: {joke_telling_frequency}
        あなたは抽象的な表現が好き？それはどんな表現？: {abstract_expression_preference}
        あなたの座右の銘: {personal_motto}
        あなたの人生で一番古い記憶: {earliest_memory}
        あなたが子供の頃に持っていた夢: {childhood_dream}
        あなたの初恋: {first_love}
        あなたを一番理解している人: {person_who_understands_you_best}
        あなたが一番安らげるの場所: {comforting_place}
        あなた自身のどの部分を一番好きというと、{favorite_physical_feature}
        あなたが他人に誤解されやすい点: {commonly_misunderstood_aspect}
        あなたがこれまでに乗り越えた最大の困難: {overcome_difficulty}
        あなたの人生において最も影響を与えた出来事: {impactful_life_event}
        あなたが人間関係で最も価値を置くの: {valued_relationship}
        あなたが大切にしている習慣、行事、宗教、伝統など: {cherished_traditions}
        あなたが自分の子供や友人に伝えたい最も大切な教訓: {important_lesson_to_share}
        あなたが全人類に伝えたいこと: {message_to_humanity}
        あなたは一日だけ超能力を手に入れるとしたら、何をしたい？: {desired_superpower}
        あなたは過去に戻れるとしたら、どの瞬間に戻りたい？: {moment_to_relive}
        あなたは過去の自分にアドバイスできるとしたら、何を伝える？: {advice_to_past_self}
        あなたにとって人生の一番の後悔?: {biggest_regret}
        あなたの人生最後に言いたい言葉は?: {final_words}
        あなたは死後、他人にどのように記憶されたい？: {desired_legacy}
        あなたはもう一度生まれ変わるなら何になりたい？: {ideal_reincarnation}

    一般的にユーザーは日本人で、日本語で話します。

    # タスク(task)
    あなたの仕事は、ユーザーの訪問目的に基づいてユーザーとチャットすることです。

    # ルール(rule)
    1. ユーザーの訪問目的とあなたとの関係に基づいて、正しく表現すること。
    2. 前の会話を続けながら、自然に説明すること。
    3. 自分がAIアシスタントであるとは言わないこと。
    4. 自分の名前や接頭辞を付けずに直接返答してください。

    # チャット履歴
    <chat_history>
    {chat_history}
    </chat_history>
    Question: {question}
    Helpful answer in markdown:
    """

    prompt_qa = ChatPromptTemplate.from_messages([
        ("system", QA_TEMPLATE),
        
    ])
    chain_qa = prompt_qa | ai_model

    print("this is history: ", chat_history)
    answer = chain_qa.invoke({
        "chat_history": chat_history,
        "question": data.get('input'),
        "name": data.get('name'),
        "user_name": data.get("user_name"),
        "relationship": data.get("relationship"),
        "visit_purpose": data.get("visit_purpose"),
        
        "birthday": data.get("birthday"),
        "hometown": data.get("hometown"),
        "gender": data.get("gender"),
        "bloodtype": data.get("bloodtype"),
        "job": data.get("job"),
        "hobby": data.get("hobby"),
        "special": data.get("special"),
        
        "family_names": data.get("family_names"),
        "self_introduction": data.get("self_introduction"),
        "first_person_pronoun": data.get("first_person_pronoun"),
        "conversation_style": data.get("conversation_style"),
        "conversation_starters": data.get("conversation_starters"),
        "conversation_endings": data.get("conversation_endings"),
        "intonation_and_volume": data.get("intonation_and_volume"),
        "conversation_pacing": data.get("conversation_pacing"),
        "fluency": data.get("fluency"),
        "dialect": data.get("dialect"),
        "languages_spoken": data.get("languages_spoken"),
        "other_speaking_styles": data.get("other_speaking_styles"),
        
        "favorite_animal": data.get("favorite_animal"),
        "favorite_book": data.get("favorite_book"),
        "favorite_movie": data.get("favorite_movie"),
        "favorite_singer_or_talent": data.get("favorite_singer_or_talent"),
        "favorite_song": data.get("favorite_song"),
        "favorite_sport": data.get("favorite_sport"),
        "favorite_game": data.get("favorite_game"),
        "favorite_artwork_or_anime": data.get("favorite_artwork_or_anime"),
        "favorite_color": data.get("favorite_color"),
        "favorite_word": data.get("favorite_word"),
        "favorite_season": data.get("favorite_season"),
        "favorite_travel_destination": data.get("favorite_travel_destination"),
        "solo_travel_or_group_travel": data.get("solo_travel_or_group_travel"),
        "favorite_and_disliked_foods_and_drinks": data.get("favorite_and_disliked_foods_and_drinks"),
        "spicy_or_sweet_food_preference": data.get("spicy_or_sweet_food_preference"),
        "cooking_skills_and_signature_dishes": data.get("cooking_skills_and_signature_dishes"),
        "favorite_possessions_or_gadgets": data.get("favorite_possessions_or_gadgets"),
        "collected_items": data.get("collected_items"),
        "most_expensive_purchase": data.get("most_expensive_purchase"),
        "favorite_purchase": data.get("favorite_purchase"),
        
        "stress_relief_methods": data.get("stress_relief_methods"),
        "definition_of_family": data.get("definition_of_family"),
        "favorite_place_at_home": data.get("favorite_place_at_home"),
        "favorite_event": data.get("favorite_event"),
        "weekend_activities": data.get("weekend_activities"),
        "morning_or_night_person": data.get("morning_or_night_person"),
        "indoor_or_outdoor_person": data.get("indoor_or_outdoor_person"),
        "active_or_laidback_person": data.get("active_or_laidback_person"),
        "optimist_or_pessimist": data.get("optimist_or_pessimist"),
        "planning_or_spontaneous_person": data.get("planning_or_spontaneous_person"),
        "recent_obsession": data.get("recent_obsession"),
        "recent_happiness": data.get("recent_happiness"),
        "recent_sadness": data.get("recent_sadness"),
        "recent_anger": data.get("recent_anger"),
        "recent_emotional_moment": data.get("recent_emotional_moment"),
        "recent_learning": data.get("recent_learning"),
        "recent_regret": data.get("recent_regret"),
        "recent_reflection": data.get("recent_reflection"),
        "recent_gratitude": data.get("recent_gratitude"),
        "recent_appreciation_received": data.get("recent_appreciation_received"),
        
        "one_word_description": data.get("one_word_description"),
        "animal_representation": data.get("animal_representation"),
        "childhood_personality": data.get("childhood_personality"),
        "dream_job_as_a_child": data.get("dream_job_as_a_child"),
        "moments_of_happiness": data.get("moments_of_happiness"),
        "point_of_pride": data.get("point_of_pride"),
        "habit_to_change": data.get("habit_to_change"),
        "memorable_feedback": data.get("memorable_feedback"),
        "life_changing_event": data.get("life_changing_event"),
        "future_challenges": data.get("future_challenges"),
        "anger_traits": data.get("anger_traits"),
        "sadness_traits": data.get("sadness_traits"),
        "lying_traits": data.get("lying_traits"),
        "struggling_traits": data.get("struggling_traits"),
        "happiness_traits": data.get("happiness_traits"),
        "laughing_style": data.get("laughing_style"),
        "crying_style": data.get("crying_style"),
        "joke_telling_frequency": data.get("joke_telling_frequency"),
        "abstract_expression_preference": data.get("abstract_expression_preference"),
        "personal_motto": data.get("personal_motto"),
        
        "earliest_memory": data.get("earliest_memory"),
        "childhood_dream": data.get("childhood_dream"),
        "first_love": data.get("first_love"),
        "person_who_understands_you_best": data.get("person_who_understands_you_best"),
        "comforting_place": data.get("comforting_place"),
        "favorite_physical_feature": data.get("favorite_physical_feature"),
        "commonly_misunderstood_aspect": data.get("commonly_misunderstood_aspect"),
        "overcome_difficulty": data.get("overcome_difficulty"),
        "impactful_life_event": data.get("impactful_life_event"),
        "valued_relationship": data.get("valued_relationship"),
        "cherished_traditions": data.get("cherished_traditions"),
        "important_lesson_to_share": data.get("important_lesson_to_share"),
        "message_to_humanity": data.get("message_to_humanity"),
        "desired_superpower": data.get("desired_superpower"),
        "moment_to_relive": data.get("moment_to_relive"),
        "advice_to_past_self": data.get("advice_to_past_self"),
        "biggest_regret": data.get("biggest_regret"),
        "final_words": data.get("final_words"),
        "desired_legacy": data.get("desired_legacy"),
        "ideal_reincarnation": data.get("ideal_reincarnation")
    })    

    chat_history.append({"Human": data.get('input')})
    chat_history.append({"AI_Model": answer.content})
    return answer.content

@app.route('/add_profile', methods=['POST'])
def add_profiles():
    create_index(index_name)
    index = pc.GRPCIndex(index_name)
    profile_data = request.json
    
    #
    for profile in profile_data:
        documents = [json.dumps(profile)]
        embeds = embed.embed_documents(documents)
        ids = [profile['id']]

        index.upsert(vectors=zip(ids, embeds))


    return "Indexed Successfully"

def create_index(index_name):
    if index_name not in pinecone.list_indexes():
        pc.create_index(
            name = index_name,
            metric = "dotproduct",
            dimension = 1536
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
