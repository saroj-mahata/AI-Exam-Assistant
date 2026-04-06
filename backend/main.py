@app.post("/ask")
async def ask_question(question: str):
    user_questions.append(question)

    content = read_all_pdfs()

    prompt = f"""
    Answer from these notes:
    
    {content}
    
    Question:
    {question}
    """

    response = model.generate_content(prompt)

    return {"answer": response.text}


@app.get("/weak-topics")
async def weak_topics():
    questions = "\n".join(user_questions)

    prompt = f"""
    Based on these user questions:
    
    {questions}
    
    Identify weak topics of the student.
    """

    response = model.generate_content(prompt)

    return {"weak_topics": response.text}
