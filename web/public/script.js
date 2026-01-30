const API_URL = 'http://localhost:3000';

let currentUser = null;

// Navigation & UI Toggles
function showLogin() {
    document.getElementById('login-form').classList.remove('hidden');
    document.getElementById('register-form').classList.add('hidden');
    document.getElementById('login-toggle').classList.add('active');
    document.getElementById('register-toggle').classList.remove('active');
    clearErrors();
}

function showRegister() {
    document.getElementById('login-form').classList.add('hidden');
    document.getElementById('register-form').classList.remove('hidden');
    document.getElementById('login-toggle').classList.remove('active');
    document.getElementById('register-toggle').classList.add('active');
    clearErrors();
}

function clearErrors() {
    document.getElementById('login-error').textContent = '';
    document.getElementById('reg-error').textContent = '';
}

function toggleDashboard() {
    const authSection = document.getElementById('auth-section');
    const dashboardSection = document.getElementById('dashboard-section');

    if (currentUser) {
        authSection.classList.add('hidden');
        dashboardSection.classList.remove('hidden');
        document.getElementById('user-greeting').textContent = `Hello, ${currentUser.username}`;
    } else {
        authSection.classList.remove('hidden');
        dashboardSection.classList.add('hidden');
    }
}

// Authentication Logic
async function handleRegister(e) {
    e.preventDefault();
    const username = document.getElementById('reg-username').value;
    const email = document.getElementById('reg-email').value;
    const password = document.getElementById('reg-password').value;
    const errorMsg = document.getElementById('reg-error');

    try {
        const response = await fetch(`${API_URL}/register`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password })
        });

        const data = await response.json();

        if (response.ok) {
            alert('Registration successful! Please login.');
            showLogin();
        } else {
            errorMsg.textContent = data.error || 'Registration failed';
        }
    } catch (error) {
        errorMsg.textContent = 'Server error. Please try again later.';
        console.error('Error:', error);
    }
}

async function handleLogin(e) {
    e.preventDefault();
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    const errorMsg = document.getElementById('login-error');

    try {
        const response = await fetch(`${API_URL}/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });

        const data = await response.json();

        if (response.ok) {
            currentUser = data.user;
            toggleDashboard();
        } else {
            errorMsg.textContent = data.error || 'Login failed';
        }
    } catch (error) {
        errorMsg.textContent = 'Server error. Please try again later.';
        console.error('Error:', error);
    }
}

function logout() {
    currentUser = null;
    toggleDashboard();
    document.getElementById('translation-box').innerHTML = '<p class="placeholder-text">Result will appear here...</p>';
}

// Dashboard Logic
function triggerUpload() {
    document.getElementById('video-upload').click();
}

function triggerRecord() {
    // For now, since it's a dummy functionality as per requirements:
    saveVideo("recorded_video.mp4");
}

async function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        saveVideo(file.name);
    }
}

async function saveVideo(videoName) {
    if (!currentUser) return;

    // UI Feedback
    const translationBox = document.getElementById('translation-box');
    translationBox.innerHTML = '<p class="placeholder-text">Processing...</p>';

    try {
        const response = await fetch(`${API_URL}/save-video`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: currentUser.id, videoName })
        });

        const data = await response.json();

        if (response.ok) {
            translationBox.innerHTML = `<p style="color: lightgreen; font-weight: 500;">${data.result}</p>`;
        } else {
            translationBox.innerHTML = `<p style="color: #ff6b6b;">Error: ${data.error}</p>`;
        }
    } catch (error) {
        translationBox.innerHTML = `<p style="color: #ff6b6b;">Server connection failed</p>`;
        console.error('Error:', error);
    }
}
