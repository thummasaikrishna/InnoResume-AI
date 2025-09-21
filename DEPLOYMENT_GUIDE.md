# Streamlit Cloud Deployment Guide

## Files Required for Deployment

1. **requirements.txt** - Contains all Python dependencies
2. **resume_system_main.py** - Main application file
3. **favicon_and_logo.jpg** - Logo file (optional)
4. **resume_analyzer.db** - Database file (if using local database)
5. **.streamlit/config.toml** - Streamlit configuration

## Deployment Steps

1. **Upload to GitHub Repository**
   - Create a new repository on GitHub
   - Upload all files to the repository
   - Make sure `requirements.txt` is in the root directory

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Connect your GitHub repository
   - Select the main file: `resume_system_main.py`
   - Click "Deploy"

## Important Notes

- The app will automatically download NLTK data and spaCy models on first run
- Make sure all dependencies are listed in `requirements.txt`
- The app uses SQLite database which will be created automatically
- File uploads are handled through Streamlit's file uploader

## Troubleshooting

If you encounter import errors:
1. Check that all packages are in `requirements.txt`
2. Ensure no built-in Python modules are listed in requirements
3. Verify package versions are compatible

## Environment Variables (if needed)

If you need to set environment variables in Streamlit Cloud:
1. Go to your app's settings
2. Add environment variables in the "Secrets" section
3. Use `st.secrets` in your code to access them
