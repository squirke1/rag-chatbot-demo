const chatMessages = document.getElementById('chatMessages');
const questionInput = document.getElementById('questionInput');
const sendButton = document.getElementById('sendButton');

function addMessage(text, isUser, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = isUser ? '👤' : '🤖';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = text;
    
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        sourcesDiv.textContent = `Sources: ${sources.join(', ')}`;
        content.appendChild(sourcesDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    chatMessages.appendChild(messageDiv);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question) return;
    
    // Add user message
    addMessage(question, true);
    questionInput.value = '';
    
    // Disable input while processing
    sendButton.disabled = true;
    questionInput.disabled = true;
    sendButton.innerHTML = '<span class="loading">Thinking</span>';
    
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                method: 'similarity'
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        addMessage(data.answer, false, data.sources);
        
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your question. Please try again.', false);
    } finally {
        sendButton.disabled = false;
        questionInput.disabled = false;
        sendButton.textContent = 'Send';
        questionInput.focus();
    }
}

// Focus input on load
questionInput.focus();
