# GitHub Pages Setup for CometAI Website

## 🌐 GitHub Pages Configuration

### Step 1: Enable GitHub Pages
1. Go to your repository: `https://github.com/thatrandomasiandev/website`
2. Click **Settings** → **Pages**
3. Source: **Deploy from a branch**
4. Branch: **main**
5. Folder: **/ (root)**
6. Save

### Step 2: Update Chat Interface for External API
The chat interface needs to be updated to connect to an external backend instead of localhost.

### Step 3: Backend Hosting Options

#### Option A: Railway (Recommended)
- **Free tier**: 500 hours/month
- **Easy deployment**: Connect GitHub repo
- **Supports**: Python, Flask, AI models
- **URL**: Your site will be at `https://your-username.github.io/website`

#### Option B: Render
- **Free tier**: 750 hours/month
- **Auto-deploy**: From GitHub
- **Supports**: Python, Flask
- **Custom domains**: Available

#### Option C: Heroku
- **Free tier**: Limited (discontinued free tier)
- **Paid plans**: Start at $7/month
- **Supports**: Python, Flask

#### Option D: Google Cloud Run
- **Free tier**: 2 million requests/month
- **Pay-per-use**: After free tier
- **Supports**: Docker containers

## 🔧 Implementation Steps

### 1. Update Chat Interface for External API
```javascript
// Change from localhost to external API
this.apiUrl = 'https://your-backend-url.railway.app/api';
```

### 2. Configure CORS on Backend
```python
# Allow GitHub Pages domain
app.config['CORS_ORIGINS'] = [
    'https://thatrandomasiandev.github.io',
    'http://localhost:8080'  # Keep for local development
]
```

### 3. Environment Variables
Set up environment variables for:
- Model path
- API keys
- Database connections

## 📁 File Structure for GitHub Pages
```
/
├── index.html              # Main website
├── chat-interface.html     # Chat interface (updated for external API)
├── assets/
│   ├── css/
│   ├── js/
│   └── images/
├── .github/workflows/      # GitHub Actions (if needed)
└── README.md
```

## 🚀 Deployment Workflow
1. **Frontend**: Auto-deploys to GitHub Pages
2. **Backend**: Deploy separately to cloud platform
3. **Domain**: Optional custom domain setup
4. **SSL**: Automatic HTTPS on both platforms

## 💰 Cost Breakdown
- **GitHub Pages**: Free
- **Railway**: Free tier (500 hours/month)
- **Total**: $0/month for basic usage
