# 🚀 OpenRouter Streamlit Chat - Alpha Beta

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io)
[![OpenRouter](https://img.shields.io/badge/API-OpenRouter-7A49FF.svg)](https://openrouter.ai)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/) <!-- Add a LICENSE file -->

**Your polished, feature-rich, and highly customizable gateway to a diverse range of LLMs via OpenRouter.ai, all within a sleek Streamlit interface!**

Experience designed for power users and developers who want fine-grained control over their AI interactions, complete with intelligent model routing, robust quota management, and a *decent* UI.

---

## ✨ Features

*   🎨 **Sleek, Modern UI:** Now includes a user-selectable dark mode.
*   🧠 **Intelligent Model Routing:** Don't just pick a model; let an AI choose the *best* model for your query, lower the cost at the price of twice the cost!
*   📊 **Comprehensive Quota Management:** Usage limits for different models to manage costs and API access effectively.
*   💾 **Persistent Chat Sessions:** Conversations are automatically saved and can be revisited. Blank, unused sessions are auto-cleaned.
*   🔄 **Real-time Streaming:** Get responses token-by-token for a fluid, ChatGPT-like experience. (This took 1 hour, this should not have taken 1 hour.)
*   💰 **Credit Monitoring:** Since I like to mention features twice, keep an eye on your OpenRouter API credits directly within the app.
*   🛠️ **Highly Configurable:** (Not uh, added - yet)
*   🛡️ **Robust Error Handling:** Gracefully handles API errors, network issues, and unexpected responses.
*   🆓 **Fallback Safety Net:** Designate a free/cost-effective model as a fallback if preferred models are unavailable or quotas are hit.
*   📝 **Detailed Logging:** Not very comprehensive logging for easy debugging and monitoring.
*   📋 **Copy-to-Clipboard:** Easily copy model responses.
---

## 🖼️ Sneak Peek

<img width="1295" alt="image" src="https://github.com/user-attachments/assets/25362495-05fe-48aa-98ec-126cc5c3402f" />

---

## 🚀 Getting Started

Follow these steps to get the OpenRouter Streamlit Chat app up and running on your local machine.

### 1. Prerequisites

*   **Python 3.8+**
*   **Git**
*   An **OpenRouter API Key** (get yours from [OpenRouter.ai](https://openrouter.ai/keys))

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 3. Install Dependencies

```bash
pip install streamlit requests
```

### 4. Run the App

```bash
streamlit run streamlit_app.py
```
