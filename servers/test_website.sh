#!/bin/bash

# Comprehensive Website Test Script
echo "🔍 Testing Personal Website Components..."

# Test server connectivity
echo "📡 Testing server connectivity..."
if curl -s -I http://localhost:8000/ | grep -q "200 OK"; then
    echo "✅ Server is running and accessible"
else
    echo "❌ Server is not accessible"
    exit 1
fi

# Test main pages
echo "📄 Testing main pages..."
pages=("index.html" "about.html" "projects.html" "skills.html" "contact.html" "resume.html" "nasa_resume.html")
for page in "${pages[@]}"; do
    if curl -s -I "http://localhost:8000/$page" | grep -q "200 OK"; then
        echo "✅ $page - OK"
    else
        echo "❌ $page - FAILED"
    fi
done

# Test critical assets
echo "🎨 Testing critical assets..."
assets=("scripts/styles.css" "scripts/script.js" "favicon.ico" "images/div/1.jpg" "images/div/2.jpg" "images/div/3.jpg" "images/div/7.jpeg")
for asset in "${assets[@]}"; do
    if curl -s -I "http://localhost:8000/$asset" | grep -q "200 OK"; then
        echo "✅ $asset - OK"
    else
        echo "❌ $asset - FAILED"
    fi
done

# Test resume PDF
echo "📋 Testing resume PDF..."
if curl -s -I "http://localhost:8000/resumes/Joshua%20Terranova%20-%20Resume.pdf" | grep -q "200 OK"; then
    echo "✅ Resume PDF - OK"
else
    echo "❌ Resume PDF - FAILED"
fi

echo "🎉 Website test complete!"
echo "🌐 Your website is accessible at: http://localhost:8000"
