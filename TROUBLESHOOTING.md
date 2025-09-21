# Streamlit Cloud Troubleshooting Guide

## Common Issues and Solutions

### 1. Requirements Installation Error

**Problem**: "Error installing requirements" message

**Solutions**:
1. **Try the minimal requirements first**:
   - Rename `requirements.txt` to `requirements_full.txt`
   - Rename `requirements_minimal.txt` to `requirements.txt`
   - Redeploy

2. **Check package compatibility**:
   - Some packages may not be compatible with Streamlit Cloud's Python version
   - Remove problematic packages one by one

3. **Memory issues**:
   - Some packages like `spacy` and `scikit-learn` are memory-intensive
   - Try removing non-essential packages first

### 2. Import Errors

**Problem**: ModuleNotFoundError after successful installation

**Solutions**:
1. **Check import statements**:
   - Ensure all imported modules are in requirements.txt
   - Remove unused imports

2. **Handle optional imports**:
   ```python
   try:
       import optional_package
   except ImportError:
       optional_package = None
   ```

### 3. File Path Issues

**Problem**: File not found errors

**Solutions**:
1. **Use relative paths**:
   - Avoid absolute paths like `C:\` or `/home/`
   - Use `Path(__file__).parent` for relative paths

2. **Handle missing files gracefully**:
   ```python
   if Path("file.txt").exists():
       # process file
   else:
       st.warning("File not found")
   ```

### 4. Database Issues

**Problem**: SQLite database errors

**Solutions**:
1. **Create database if it doesn't exist**:
   ```python
   if not Path("database.db").exists():
       # create database
   ```

2. **Use in-memory database for testing**:
   ```python
   conn = sqlite3.connect(":memory:")
   ```

## Step-by-Step Debugging

1. **Start with minimal requirements**
2. **Test locally first** with `streamlit run resume_system_main.py`
3. **Add packages one by one** to identify problematic ones
4. **Check Streamlit Cloud logs** for specific error messages
5. **Use try-except blocks** for optional features

## Alternative Deployment Options

If Streamlit Cloud continues to have issues:

1. **Heroku**: More control over environment
2. **Railway**: Simple deployment with better package support
3. **Render**: Free tier with good Python support
4. **Local deployment**: Run on your own server
