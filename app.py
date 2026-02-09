# app.py
# Streamlit: AI 습관 트래커 📊
# 실행: streamlit run app.py

import os
import re
import json
import time
import calendar
from datetime import datetime, timedelta, date

import streamlit as st
import pandas as pd
import requests

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")


# -----------------------------
# Helpers: session state init
# -----------------------------
HABITS = [
    ("wake", "🌅", "기상 미션"),
    ("water", "💧", "물 마시기"),
    ("study", "📚", "공부/독서"),
    ("workout", "🏃", "운동하기"),
    ("sleep", "😴", "수면"),
]

CITIES = [
    "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
    "Gwangju", "Ulsan", "Suwon", "Sejong", "Jeju"
]

COACH_STYLES = {
    "스파르타 코치": "엄격하고 직설적이며 실행을 강하게 요구하는 코치. 변명 차단, 구체적 지시, 단호한 톤.",
    "따뜻한 멘토": "공감과 격려 중심. 현실적인 조언과 작은 성공을 칭찬. 부드럽고 따뜻한 톤.",
    "게임 마스터": "RPG 퀘스트/레벨업 스타일. 재미있고 몰입감 있게. 용어: 퀘스트, 경험치, 보상, 보스전 등.",
}

def _today_str() -> str:
    return date.today().isoformat()

def _date_str(target_date: date) -> str:
    return target_date.isoformat()

def _calc_achievement(habit_state: dict) -> tuple[int, int, float]:
    done = sum(1 for k, _, _ in HABITS if habit_state.get(k, False))
    total = len(HABITS)
    rate = round((done / total) * 100, 1)
    return done, total, rate

def _init_demo_history():
    """6일 샘플 + 오늘은 실시간 입력으로 합쳐서 7일 차트를 만들기 위함."""
    today = date.today()
    demo = []
    # 6일치 (today-6 ~ today-1)
    # 너무 랜덤하면 UX가 들쭉날쭉해서, 패턴이 보이는 샘플로 구성
    pattern = [
        (2, 5), (3, 6), (4, 7), (3, 6), (5, 8), (4, 7)
    ]
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        done_cnt, mood = pattern[6 - i]
        habit_keys = [k for k, _, _ in HABITS][:done_cnt]
        habit_state = {k: (k in habit_keys) for k, _, _ in HABITS}
        demo.append({
            "date": d.isoformat(),
            "done": done_cnt,
            "rate": round(done_cnt / len(HABITS) * 100, 1),
            "mood": mood,
            "habits": habit_state,
        })
    return demo

def _get_history_record(target_date: date) -> dict | None:
    target_str = _date_str(target_date)
    for record in st.session_state.history:
        if record.get("date") == target_str:
            return record
    return None

def _apply_record_to_state(target_date: date):
    record = _get_history_record(target_date)
    if record:
        habits = record.get("habits", {})
        for k, _, _ in HABITS:
            st.session_state[f"habit_{k}"] = habits.get(k, False)
        st.session_state.mood_slider = record.get("mood", 7)
    else:
        for k, _, _ in HABITS:
            st.session_state[f"habit_{k}"] = False
        st.session_state.mood_slider = 7

if "history" not in st.session_state:
    st.session_state.history = _init_demo_history()

if "last_report" not in st.session_state:
    st.session_state.last_report = ""

if "last_share_text" not in st.session_state:
    st.session_state.last_share_text = ""

if "last_weather" not in st.session_state:
    st.session_state.last_weather = None

if "last_dog" not in st.session_state:
    st.session_state.last_dog = None

if "last_quote" not in st.session_state:
    st.session_state.last_quote = None

if "last_advice" not in st.session_state:
    st.session_state.last_advice = None

if "last_sun_times" not in st.session_state:
    st.session_state.last_sun_times = None

if "checkin_date" not in st.session_state:
    st.session_state.checkin_date = date.today()

if "habit_initialized" not in st.session_state:
    _apply_record_to_state(st.session_state.checkin_date)
    st.session_state.habit_initialized = True


# -----------------------------
# API: Weather / Dog / Quote / Advice / Sun Times
# -----------------------------
def get_weather(city: str, api_key: str):
    """
    OpenWeatherMap에서 날씨 가져오기 (한국어, 섭씨)
    실패 시 None 반환, timeout=10
    """
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",
            "lang": "kr",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "city": city,
            "temp_c": data.get("main", {}).get("temp"),
            "feels_like_c": data.get("main", {}).get("feels_like"),
            "humidity": data.get("main", {}).get("humidity"),
            "desc": (data.get("weather") or [{}])[0].get("description"),
            "icon": (data.get("weather") or [{}])[0].get("icon"),
            "lat": data.get("coord", {}).get("lat"),
            "lon": data.get("coord", {}).get("lon"),
        }
    except Exception:
        return None


def _parse_breed_from_image_url(image_url: str) -> str:
    """
    Dog CEO 이미지 URL 예:
    https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
    -> hound-afghan (sub-breed 포함 가능)
    """
    try:
        m = re.search(r"/breeds/([^/]+)/", image_url)
        if not m:
            return "알 수 없음"
        raw = m.group(1).strip()
        # 보기 좋게 변환: "hound-afghan" -> "hound (afghan)"
        parts = raw.split("-")
        if len(parts) >= 2:
            return f"{parts[0]} ({' '.join(parts[1:])})"
        return raw
    except Exception:
        return "알 수 없음"


def get_dog_image():
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종 가져오기
    실패 시 None 반환, timeout=10
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None
        img = data.get("message")
        if not img:
            return None
        breed = _parse_breed_from_image_url(img)
        return {"image_url": img, "breed": breed}
    except Exception:
        return None

def get_quote():
    """Quotable에서 랜덤 명언 가져오기."""
    try:
        url = "https://api.quotable.io/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "text": data.get("content"),
            "author": data.get("author"),
        }
    except Exception:
        return None

def get_advice():
    """Advice Slip에서 랜덤 조언 가져오기."""
    try:
        url = "https://api.adviceslip.com/advice"
        r = requests.get(url, timeout=10, headers={"Accept": "application/json"})
        if r.status_code != 200:
            return None
        data = r.json()
        slip = data.get("slip", {})
        return {"text": slip.get("advice")}
    except Exception:
        return None

def get_sun_times(lat: float | None, lon: float | None):
    """Sunrise-Sunset API로 일출/일몰 가져오기."""
    if lat is None or lon is None:
        return None
    try:
        url = "https://api.sunrise-sunset.org/json"
        params = {"lat": lat, "lng": lon, "formatted": 0}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "OK":
            return None
        results = data.get("results", {})
        sunrise_raw = results.get("sunrise")
        sunset_raw = results.get("sunset")
        if not sunrise_raw or not sunset_raw:
            return None
        sunrise = datetime.fromisoformat(sunrise_raw.replace("Z", "+00:00")).astimezone()
        sunset = datetime.fromisoformat(sunset_raw.replace("Z", "+00:00")).astimezone()
        return {
            "sunrise": sunrise.strftime("%H:%M"),
            "sunset": sunset.strftime("%H:%M"),
            "day_length": results.get("day_length"),
        }
    except Exception:
        return None


# -----------------------------
# AI: Report generation
# -----------------------------
def _build_system_prompt(coach_style: str) -> str:
    base = COACH_STYLES.get(coach_style, COACH_STYLES["따뜻한 멘토"])
    format_rules = """
너는 사용자의 '습관 체크인' 데이터를 기반으로 짧고 강력한 코칭 리포트를 작성한다.
반드시 아래 출력 형식을 지켜라(섹션 제목 포함, 순서 고정):

[컨디션 등급] S|A|B|C|D 중 1개
[습관 분석] (짧은 문단 + 핵심 불릿 2~4개)
[날씨 코멘트] (한 문단)
[내일 미션] (불릿 3개, 구체적/측정 가능)
[오늘의 한마디] (한 줄, 감정/동기 부여)

주의:
- 과장 금지. 입력값을 근거로 평가.
- 유저를 비난하지 말되, 스타일에 맞게 톤을 조절.
- 한국어로 작성.
- 명언/조언/일출·일몰 정보가 있으면 자연스럽게 한두 줄 반영.
"""
    # 스타일별 강화 지침
    style_add = ""
    if coach_style == "스파르타 코치":
        style_add = """
톤: 단호/직설/군더더기 없음. 핑계 차단. 행동 지시를 명령형으로.
"""
    elif coach_style == "따뜻한 멘토":
        style_add = """
톤: 따뜻/공감/격려. 작은 성취를 칭찬하고, 실패는 부담 없이 재설계.
"""
    elif coach_style == "게임 마스터":
        style_add = """
톤: RPG 게임 진행자. 경험치/퀘스트/레벨업/보상 용어를 자연스럽게 섞어 재미있게.
"""
    return f"{base}\n{style_add}\n{format_rules}".strip()


def generate_report(
    openai_api_key: str,
    coach_style: str,
    habit_state: dict,
    mood: int,
    weather: dict | None,
    dog: dict | None,
):
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달
    모델: gpt-5-mini
    실패 시 None 반환
    """
    if not openai_api_key:
        return None

    # OpenAI SDK 사용 (권장)
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return {
            "error": "OpenAI Python 라이브러리(openai)가 없습니다. `pip install openai` 후 다시 실행하세요."
        }

    done, total, rate = _calc_achievement(habit_state)

    habits_done_list = [label for k, _, label in HABITS if habit_state.get(k, False)]
    habits_miss_list = [label for k, _, label in HABITS if not habit_state.get(k, False)]

    weather_txt = "날씨 정보 없음"
    if weather:
        weather_txt = (
            f"{weather.get('city')} / {weather.get('desc')} / "
            f"{weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C) / 습도 {weather.get('humidity')}%"
        )

    dog_breed = dog.get("breed") if dog else "랜덤 강아지 정보 없음"

    user_payload = {
        "date": _date_str(st.session_state.checkin_date),
        "habits_done": habits_done_list,
        "habits_missed": habits_miss_list,
        "done_count": done,
        "total_habits": total,
        "achievement_rate_percent": rate,
        "mood_1_to_10": mood,
        "weather": weather_txt,
        "dog_breed": dog_breed,
        "quote": st.session_state.last_quote,
        "advice": st.session_state.last_advice,
        "sun_times": st.session_state.last_sun_times,
    }

    user_message = f"""
아래 체크인 데이터를 기반으로 리포트를 작성해줘.
데이터(JSON):
{json.dumps(user_payload, ensure_ascii=False, indent=2)}
""".strip()

    try:
        client = OpenAI(api_key=openai_api_key)
        resp = client.responses.create(
            model="gpt-5-mini",
            instructions=_build_system_prompt(coach_style),
            input=user_message,
        )
        text = getattr(resp, "output_text", None) or ""
        text = text.strip()
        if not text:
            return None
        return {"text": text, "payload": user_payload}
    except Exception as e:
        return {"error": f"OpenAI 호출 실패: {e}"}


# -----------------------------
# Sidebar: API keys
# -----------------------------
with st.sidebar:
    st.header("🔑 API Key 설정")
    openai_key = st.text_input("OpenAI API Key", type="password", help="OpenAI API 키를 입력하세요.")
    owm_key = st.text_input("OpenWeatherMap API Key", type="password", help="OpenWeatherMap API 키를 입력하세요.")
    st.caption("키는 session_state에만 유지되고 파일로 저장하지 않습니다.")


# -----------------------------
# Main UI: Habit check-in
# -----------------------------
st.title("📊 AI 습관 트래커")

left, right = st.columns([1.05, 1])

with left:
    st.subheader("✅ 오늘의 습관 체크인")
    st.date_input(
        "📅 체크인 날짜",
        value=st.session_state.checkin_date,
        key="checkin_date",
        on_change=lambda: _apply_record_to_state(st.session_state.checkin_date),
    )

    c1, c2 = st.columns(2)
    habit_state = {}

    # 2열 배치: 왼쪽 3개 / 오른쪽 2개
    with c1:
        for k, emoji, label in HABITS[:3]:
            habit_state[k] = st.checkbox(f"{emoji} {label}", key=f"habit_{k}")
    with c2:
        for k, emoji, label in HABITS[3:]:
            habit_state[k] = st.checkbox(f"{emoji} {label}", key=f"habit_{k}")

    mood = st.slider("😊 오늘 기분 점수", min_value=1, max_value=10, value=7, key="mood_slider")

    city = st.selectbox("🏙️ 도시 선택", options=CITIES, index=0, key="city_select")
    coach_style = st.radio("🎭 코치 스타일", options=list(COACH_STYLES.keys()), horizontal=True, key="coach_style")


# -----------------------------
# Achievement + Metrics + Chart
# -----------------------------
done, total, rate = _calc_achievement(habit_state)

with right:
    st.subheader("📈 달성률 & 주간 추이")

    m1, m2, m3 = st.columns(3)
    m1.metric("달성률", f"{rate} %")
    m2.metric("달성 습관", f"{done} / {total}")
    m3.metric("기분", f"{mood} / 10")

    # 오늘 데이터를 히스토리에 "가상 반영"해서 7일 차트 생성 (실제 저장은 리포트 생성 시 업서트)
    history = list(st.session_state.history)
    target_date_str = _date_str(st.session_state.checkin_date)
    today_record = {"date": target_date_str, "done": done, "rate": rate, "mood": mood}
    history = [r for r in history if r.get("date") != target_date_str] + [today_record]
    history_sorted = sorted(history, key=lambda x: x["date"])[-7:]

    df = pd.DataFrame(history_sorted)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%m/%d")
    df = df.set_index("date")

    st.caption("최근 7일 달성률(%)")
    st.bar_chart(df[["rate"]], height=260)


# -----------------------------
# Action: Generate report
# -----------------------------
st.divider()

btn_col1, btn_col2 = st.columns([1, 2])
with btn_col1:
    generate = st.button("🧠 컨디션 리포트 생성", type="primary", use_container_width=True)

def _upsert_today_history(done_cnt: int, rate_pct: float, mood_score: int):
    target_date = st.session_state.checkin_date
    habit_payload = {k: habit_state.get(k, False) for k, _, _ in HABITS}
    rec = {
        "date": _date_str(target_date),
        "done": done_cnt,
        "rate": rate_pct,
        "mood": mood_score,
        "habits": habit_payload,
    }
    st.session_state.history = [r for r in st.session_state.history if r.get("date") != rec["date"]] + [rec]
    st.session_state.history = sorted(st.session_state.history, key=lambda x: x["date"])[-90:]

if generate:
    # 1) 기록 저장
    _upsert_today_history(done, rate, mood)

    # 2) 외부 API 호출
    weather = get_weather(city, owm_key)
    dog = get_dog_image()
    quote = get_quote()
    advice = get_advice()
    sun_times = get_sun_times(
        weather.get("lat") if weather else None,
        weather.get("lon") if weather else None,
    )

    st.session_state.last_weather = weather
    st.session_state.last_dog = dog
    st.session_state.last_quote = quote
    st.session_state.last_advice = advice
    st.session_state.last_sun_times = sun_times

    # 3) OpenAI 리포트 생성
    result = generate_report(
        openai_api_key=openai_key,
        coach_style=coach_style,
        habit_state=habit_state,
        mood=mood,
        weather=weather,
        dog=dog,
    )

    if result is None:
        st.error("리포트를 생성하지 못했어요. (빈 응답)")
    elif "error" in result:
        st.error(result["error"])
    else:
        report_text = result["text"]
        payload = result["payload"]

        # 공유용 텍스트
        share_text = f"""AI 습관 트래커 리포트 ({payload["date"]})

달성률: {payload["achievement_rate_percent"]}%
달성: {", ".join(payload["habits_done"]) if payload["habits_done"] else "없음"}
미달성: {", ".join(payload["habits_missed"]) if payload["habits_missed"] else "없음"}
기분: {payload["mood_1_to_10"]}/10
날씨: {payload["weather"]}
강아지: {payload["dog_breed"]}
명언: {(payload.get("quote") or {}).get("text") if payload.get("quote") else "없음"}
조언: {(payload.get("advice") or {}).get("text") if payload.get("advice") else "없음"}
일출/일몰: {(payload.get("sun_times") or {}).get("sunrise") if payload.get("sun_times") else "없음"} / {(payload.get("sun_times") or {}).get("sunset") if payload.get("sun_times") else "없음"}

{report_text}
""".strip()

        st.session_state.last_report = report_text
        st.session_state.last_share_text = share_text


# -----------------------------
# Results display (weather + dog + report)
# -----------------------------
if st.session_state.last_report:
    st.subheader(f"🧾 {st.session_state.checkin_date.strftime('%Y-%m-%d')} 결과")

    st.markdown("#### 🌤️ 데일리 브리핑")
    brief_cols = st.columns(3)
    with brief_cols[0]:
        st.markdown("**🗣️ 명언**")
        if st.session_state.last_quote:
            st.write(st.session_state.last_quote.get("text"))
            st.caption(f"- {st.session_state.last_quote.get('author', 'Unknown')}")
        else:
            st.info("명언을 불러오지 못했어요.")
    with brief_cols[1]:
        st.markdown("**💡 오늘의 조언**")
        if st.session_state.last_advice:
            st.write(st.session_state.last_advice.get("text"))
        else:
            st.info("조언을 불러오지 못했어요.")
    with brief_cols[2]:
        st.markdown("**🌅 일출/일몰**")
        if st.session_state.last_sun_times:
            st.write(f"일출: {st.session_state.last_sun_times.get('sunrise')}")
            st.write(f"일몰: {st.session_state.last_sun_times.get('sunset')}")
            st.caption(f"일장: {st.session_state.last_sun_times.get('day_length')}")
        else:
            st.info("일출/일몰 정보를 불러오지 못했어요.")

    cA, cB = st.columns(2)

    # Weather card
    with cA:
        st.markdown("#### 🌦️ 날씨")
        w = st.session_state.last_weather
        if w is None:
            st.info("날씨 정보를 불러오지 못했어요. (API Key/도시/네트워크 확인)")
        else:
            icon = w.get("icon")
            icon_url = f"https://openweathermap.org/img/wn/{icon}@2x.png" if icon else None
            if icon_url:
                st.image(icon_url, width=80)
            st.write(f"**도시:** {w.get('city')}")
            st.write(f"**상태:** {w.get('desc')}")
            st.write(f"**기온:** {w.get('temp_c')}°C (체감 {w.get('feels_like_c')}°C)")
            st.write(f"**습도:** {w.get('humidity')}%")

    # Dog card
    with cB:
        st.markdown("#### 🐶 오늘의 강아지")
        d = st.session_state.last_dog
        if d is None:
            st.info("강아지 이미지를 불러오지 못했어요. (네트워크 확인)")
        else:
            st.image(d["image_url"], use_container_width=True)
            st.caption(f"품종: {d.get('breed', '알 수 없음')}")

    st.markdown("#### 🤖 AI 코치 리포트")
    st.write(st.session_state.last_report)

    st.markdown("#### 🔗 공유용 텍스트")
    st.code(st.session_state.last_share_text, language="text")


# -----------------------------
# Calendar View
# -----------------------------
st.divider()
st.subheader("📅 습관 캘린더")

history_map = {r.get("date"): r for r in st.session_state.history}

def _calendar_badge(rate_value: float | None) -> str:
    if rate_value is None:
        return "·"
    if rate_value >= 80:
        return "🌟"
    if rate_value >= 50:
        return "🙂"
    if rate_value > 0:
        return "🫧"
    return "⚪"

month_options = [
    (date.today().replace(day=1) - timedelta(days=30 * i)).replace(day=1)
    for i in range(0, 6)
]
month_labels = [m.strftime("%Y-%m") for m in month_options]
selected_month_label = st.selectbox("보기 월 선택", options=month_labels, index=0)
selected_month = month_options[month_labels.index(selected_month_label)]

st.caption("🌟 80% 이상 · 🙂 50% 이상 · 🫧 1~49% · ⚪ 0%")

cal = calendar.Calendar(firstweekday=0)
weeks = cal.monthdayscalendar(selected_month.year, selected_month.month)

weekday_labels = ["월", "화", "수", "목", "금", "토", "일"]
header_cols = st.columns(7)
for idx, label in enumerate(weekday_labels):
    header_cols[idx].markdown(f"**{label}**")

for week in weeks:
    day_cols = st.columns(7)
    for idx, day_num in enumerate(week):
        if day_num == 0:
            day_cols[idx].markdown(" ")
            continue
        day_date = date(selected_month.year, selected_month.month, day_num)
        record = history_map.get(_date_str(day_date))
        rate_value = record.get("rate") if record else None
        badge = _calendar_badge(rate_value)
        rate_text = f"{rate_value}%" if rate_value is not None else "-"
        day_cols[idx].markdown(f"**{day_num}**")
        day_cols[idx].caption(f"{badge} {rate_text}")


# -----------------------------
# API 안내
# -----------------------------
with st.expander("📌 API 안내 / 문제 해결", expanded=False):
    st.markdown(
        """
- **OpenAI API Key**: OpenAI 플랫폼에서 발급한 키를 사이드바에 입력하세요.
- **OpenWeatherMap API Key**: OpenWeatherMap에서 발급한 키를 사이드바에 입력하세요.
- 날씨가 `None`으로 나오면:
  - 키가 올바른지 / 도시 이름이 맞는지 / 무료 플랜 호출 제한인지 확인하세요.
- OpenAI 오류가 나면:
  - 키 유효성, 결제/쿼터, 네트워크, 그리고 `pip install openai` 설치 여부를 확인하세요.
- 이 앱은 데모용으로 **session_state**에만 저장합니다(브라우저 새로고침 시 초기화될 수 있어요).
- 추가 API: Quotable(명언), Advice Slip(조언), Sunrise-Sunset(일출/일몰)
        """.strip()
    )
