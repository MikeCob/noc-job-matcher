"""
NOC Job Matcher - Web Application
Match job descriptions to NOC codes using semantic search
"""

import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from collections import defaultdict
import html

# Page configuration
st.set_page_config(
    page_title="NOC Job Matcher",
    page_icon="💼",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .match-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #0066cc;
    }
    .score-badge {
        background-color: #0066cc;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache model and data loading
@st.cache_resource
def load_model():
    """Load the sentence transformer model - upgraded for better accuracy"""
    return SentenceTransformer('all-mpnet-base-v2')

@st.cache_data
def load_noc_data():
    """Load NOC embeddings and metadata"""
    embeddings = np.load('noc_embeddings.npy')
    duty_embeddings = np.load('duty_embeddings.npy')
    with open('noc_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return embeddings, duty_embeddings, metadata

def extract_keywords(job_description, top_n=20):
    """Extract key terms from job description"""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', job_description.lower())
    stopwords = {'the', 'and', 'for', 'are', 'this', 'that', 'with', 'from', 'will', 
                 'have', 'has', 'can', 'our', 'you', 'your', 'their', 'they', 'been',
                 'also', 'such', 'other', 'into', 'more', 'than', 'some', 'about'}
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    return list(set(keywords))

def highlight_matches(text, keywords):
    """Highlight matching keywords in text"""
    if not text or not keywords:
        return str(text) if text else ""
    
    text_str = str(text)
    
    # Avoid double-highlighting - check for mark tags (both raw and escaped)
    if ('<mark' in text_str or '&lt;mark' in text_str):
        return text_str
    
    # Apply highlighting directly (no escaping needed since NOC data is from safe source)
    highlighted = text_str
    for keyword in keywords:
        if len(keyword) < 4:
            continue
        # Match keywords case-insensitively
        pattern = re.compile(r'\b(' + re.escape(keyword) + r')\b', re.IGNORECASE)
        highlighted = pattern.sub(r'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;">\1</mark>', highlighted)
    
    return highlighted

def extract_responsibilities(job_description):
    """Extract responsibility sentences from job description"""
    # Split into sentences
    sentences = re.split(r'[.!?\n]+', job_description)
    
    # Filter for responsibility-like sentences
    responsibility_keywords = [
        'develop', 'manage', 'create', 'implement', 'design', 'coordinate',
        'lead', 'supervise', 'analyze', 'maintain', 'ensure', 'provide',
        'support', 'review', 'prepare', 'conduct', 'monitor', 'plan',
        'organize', 'direct', 'control', 'evaluate', 'establish', 'perform'
    ]
    
    responsibilities = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
        # Check if sentence contains responsibility keywords
        if any(keyword in sentence.lower() for keyword in responsibility_keywords):
            responsibilities.append(sentence)
        # Also include sentences that look like bullet points or duties
        elif sentence and (sentence[0].isupper() or sentence.startswith('-')):
            responsibilities.append(sentence.lstrip('- •'))
    
    # If no responsibilities found, use all sentences
    if not responsibilities:
        responsibilities = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    return responsibilities[:20]  # Limit to top 20
    """Extract responsibility sentences from job description"""
    # Split into sentences
    sentences = re.split(r'[.!?\n]+', job_description)
    
    # Filter for responsibility-like sentences
    responsibility_keywords = [
        'develop', 'manage', 'create', 'implement', 'design', 'coordinate',
        'lead', 'supervise', 'analyze', 'maintain', 'ensure', 'provide',
        'support', 'review', 'prepare', 'conduct', 'monitor', 'plan',
        'organize', 'direct', 'control', 'evaluate', 'establish', 'perform'
    ]
    
    responsibilities = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
        # Check if sentence contains responsibility keywords
        if any(keyword in sentence.lower() for keyword in responsibility_keywords):
            responsibilities.append(sentence)
        # Also include sentences that look like bullet points or duties
        elif sentence and (sentence[0].isupper() or sentence.startswith('-')):
            responsibilities.append(sentence.lstrip('- •'))
    
    # If no responsibilities found, use all sentences
    if not responsibilities:
        responsibilities = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    return responsibilities[:20]  # Limit to top 20

def match_duties_to_responsibilities(job_responsibilities, duty_embeddings, metadata, model):
    """Match job responsibilities to specific NOC duties"""
    if not job_responsibilities:
        return {}
    
    # Encode job responsibilities
    resp_embeddings = model.encode(job_responsibilities)
    
    # Calculate similarity between each responsibility and each duty
    similarities = cosine_similarity(resp_embeddings, duty_embeddings)
    
    # For each NOC, aggregate duty match scores
    noc_duty_scores = defaultdict(list)
    
    for duty_idx in range(len(metadata['all_duties'])):
        noc_idx = metadata['duty_to_noc_map'][duty_idx]
        duty_text = metadata['all_duties'][duty_idx]
        
        # Get best match score for this duty across all responsibilities
        best_score = similarities[:, duty_idx].max()
        best_resp_idx = similarities[:, duty_idx].argmax()
        
        if best_score > 0.3:  # Threshold for relevance
            noc_duty_scores[noc_idx].append({
                'duty': duty_text,
                'score': best_score,
                'matched_responsibility': job_responsibilities[best_resp_idx]
            })
    
    return noc_duty_scores

def find_matching_nocs(job_description, model, embeddings, duty_embeddings, metadata, top_k=10):
    """Find top matching NOC codes using hybrid approach"""
    
    # Extract responsibilities from job description
    job_responsibilities = extract_responsibilities(job_description)
    
    # Method 1: Overall semantic similarity (40% weight)
    job_embedding = model.encode([job_description])
    overall_similarities = cosine_similarity(job_embedding, embeddings)[0]
    
    # Method 2: Duty-by-duty matching (60% weight)
    duty_scores = match_duties_to_responsibilities(
        job_responsibilities, duty_embeddings, metadata, model
    )
    
    # Calculate combined scores
    combined_scores = []
    for idx in range(len(metadata['noc_codes'])):
        overall_score = overall_similarities[idx] * 0.4
        
        # Calculate duty match score
        if idx in duty_scores and duty_scores[idx]:
            # Average of top matched duties
            top_duty_scores = sorted([d['score'] for d in duty_scores[idx]], reverse=True)[:5]
            duty_score = np.mean(top_duty_scores) * 0.6
        else:
            duty_score = 0
        
        combined_score = overall_score + duty_score
        combined_scores.append(combined_score)
    
    # Get top K matches
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    # Extract keywords for highlighting
    keywords = extract_keywords(job_description)
    
    results = []
    for idx in top_indices:
        matched_duties = duty_scores.get(idx, [])
        matched_duties_sorted = sorted(matched_duties, key=lambda x: x['score'], reverse=True)
        
        results.append({
            'noc_code': metadata['noc_codes'][idx],
            'title': metadata['titles'][idx],
            'description': metadata['descriptions'][idx],
            'main_duties': metadata['main_duties'][idx],
            'example_titles': metadata['example_titles'][idx],
            'employment_requirements': metadata['employment_requirements'][idx],
            'additional_information': metadata['additional_information'][idx],
            'exclusions': metadata['exclusions'][idx],
            'url': metadata['urls'][idx],
            'overall_score': overall_similarities[idx],
            'duty_match_score': np.mean([d['score'] for d in matched_duties]) if matched_duties else 0,
            'similarity_score': combined_scores[idx],
            'matched_duties': matched_duties_sorted,
            'keywords': keywords
        })
    
    return results

# Main app
def main():
    # Header
    st.title("💼 NOC Job Matcher")
    st.markdown("### Find the best National Occupational Classification (NOC) codes for any job description")
    st.markdown("---")
    
    # Check if embeddings exist, if not generate them
    import os
    if not os.path.exists('noc_embeddings.npy') or not os.path.exists('duty_embeddings.npy'):
        st.warning("⏳ First-time setup: Generating AI embeddings... This will take 5-7 minutes.")
        st.info("The app will automatically reload when ready. Please wait...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Import preparation functions
            status_text.text("Loading NOC data...")
            progress_bar.progress(10)
            
            df = pd.read_csv('noc_data_full.csv')
            
            # Parse list fields
            def parse_list_field(x):
                if pd.isna(x) or x == '' or x == 'nan':
                    return []
                x_str = str(x).strip()
                if x_str.startswith('['):
                    try:
                        return eval(x_str)
                    except:
                        return []
                elif '|' in x_str:
                    return [item.strip() for item in x_str.split('|') if item.strip()]
                else:
                    return []
            
            df['main_duties_list'] = df['main_duties'].apply(parse_list_field)
            df['example_titles_list'] = df['example_titles'].apply(parse_list_field)
            df['exclusions_list'] = df['exclusions'].apply(parse_list_field)
            
            progress_bar.progress(20)
            status_text.text("Creating searchable text...")
            
            # Create searchable text
            def create_searchable_text(row):
                parts = []
                parts.append(f"Title: {row['title']} {row['title']}")
                parts.append(f"Description: {row['description']}")
                parts.append(row['description'][:200])
                
                if row['main_duties_list']:
                    main_duties_text = " ".join(row['main_duties_list'])
                    parts.append(f"Main duties: {main_duties_text}")
                    parts.append(f"Responsibilities: {main_duties_text}")
                    parts.append(f"Key duties: {main_duties_text}")
                
                if row['example_titles_list']:
                    parts.append("Example titles: " + " ".join(row['example_titles_list']))
                
                if pd.notna(row['employment_requirements']) and row['employment_requirements']:
                    parts.append(f"Requirements: {row['employment_requirements']}")
                
                if pd.notna(row['additional_information']) and row['additional_information']:
                    parts.append(str(row['additional_information'])[:100])
                
                if row['exclusions_list']:
                    parts.append("Exclusions: " + " ".join(row['exclusions_list'][:3]))
                
                if pd.notna(row['broad_category']) and row['broad_category']:
                    parts.append(f"Category: {row['broad_category']}")
                if pd.notna(row['major_group']) and row['major_group']:
                    parts.append(f"Group: {row['major_group']}")
                
                return " ".join(filter(None, parts))
            
            df['searchable_text'] = df.apply(create_searchable_text, axis=1)
            
            progress_bar.progress(30)
            status_text.text("Loading AI model (this may take a minute)...")
            
            model = SentenceTransformer('all-mpnet-base-v2')
            
            progress_bar.progress(40)
            status_text.text(f"Generating profile embeddings for {len(df)} NOC codes...")
            
            embeddings = model.encode(df['searchable_text'].tolist(), batch_size=32, show_progress_bar=False)
            
            progress_bar.progress(60)
            status_text.text("Generating duty-level embeddings...")
            
            all_duties = []
            duty_to_noc_map = []
            
            for idx, row in df.iterrows():
                if row['main_duties_list']:
                    for duty in row['main_duties_list']:
                        if duty and duty.strip():
                            all_duties.append(duty)
                            duty_to_noc_map.append(idx)
            
            duty_embeddings = model.encode(all_duties, batch_size=32, show_progress_bar=False)
            
            progress_bar.progress(80)
            status_text.text("Saving embeddings...")
            
            np.save('noc_embeddings.npy', embeddings)
            np.save('duty_embeddings.npy', duty_embeddings)
            
            metadata = {
                'noc_codes': df['noc_code'].tolist(),
                'titles': df['title'].tolist(),
                'descriptions': df['description'].tolist(),
                'main_duties': df['main_duties_list'].tolist(),
                'example_titles': df['example_titles_list'].tolist(),
                'employment_requirements': df['employment_requirements'].tolist(),
                'additional_information': df['additional_information'].tolist(),
                'exclusions': df['exclusions_list'].tolist(),
                'urls': df['url'].tolist(),
                'searchable_texts': df['searchable_text'].tolist(),
                'all_duties': all_duties,
                'duty_to_noc_map': duty_to_noc_map
            }
            
            with open('noc_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
            
            progress_bar.progress(100)
            status_text.text("✅ Embeddings generated successfully!")
            
            st.success(f"Generated {len(embeddings)} profile embeddings and {len(duty_embeddings)} duty embeddings!")
            st.info("Reloading app...")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating embeddings: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        top_k = st.slider("Number of results to show", min_value=1, max_value=20, value=10)
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This app uses **advanced semantic matching** with:
        - 🧠 **MPNet AI model** for understanding context
        - 🎯 **Duty-by-duty comparison** for precision
        - 📊 **Hybrid scoring** (40% overall + 60% specific duties)
        
        **How to use:**
        1. Paste or type a job description
        2. Click "Find Matching NOCs"
        3. Review matches with duty-level scores
        
        **Data:** NOC 2021 Version 1.0 (516 Unit Groups)
        """)
    
    # Load model and data
    try:
        with st.spinner("Loading AI model and NOC database..."):
            model = load_model()
            embeddings, duty_embeddings, metadata = load_noc_data()
        
        st.success(f"✅ Loaded {len(metadata['noc_codes'])} NOC codes with {len(metadata['all_duties'])} individual duties")
        
    except FileNotFoundError as e:
        st.error(f"""
        ❌ **Embeddings not found!**
        
        The app needs to generate embeddings first. Please refresh the page.
        
        Error details: {str(e)}
        """)
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("📝 Job Description Input")
        job_description = st.text_area(
            "Paste the job description here:",
            height=300,
            placeholder="Example: We are looking for a Senior Software Engineer to design and develop web applications. "
                       "Responsibilities include writing clean code, reviewing pull requests, mentoring junior developers, "
                       "and collaborating with product teams to deliver high-quality software solutions..."
        )
        
        search_button = st.button("🔍 Find Matching NOCs", type="primary")
    
    with col2:
        st.header("💡 Sample Job Descriptions")
        
        samples = {
            "Software Developer": """We are seeking a talented Software Developer to join our team. 
            Responsibilities: Design and develop software applications, write clean and efficient code, 
            perform code reviews, debug and fix software defects, collaborate with cross-functional teams, 
            participate in agile development processes, and maintain technical documentation.""",
            
            "Registered Nurse": """Looking for a compassionate Registered Nurse. Duties include: 
            Assess patient conditions, administer medications and treatments, monitor patient vital signs, 
            collaborate with physicians and healthcare team, maintain patient records, provide patient education, 
            ensure patient safety and comfort, follow infection control protocols.""",
            
            "Marketing Manager": """Seeking an experienced Marketing Manager. Responsibilities: 
            Develop and implement marketing strategies, manage marketing campaigns, analyze market trends, 
            oversee social media presence, manage marketing budget, coordinate with sales teams, 
            monitor campaign performance, conduct market research, and supervise marketing staff."""
        }
        
        for job_title, sample_desc in samples.items():
            if st.button(f"📋 {job_title}"):
                st.session_state['job_description'] = sample_desc
                st.rerun()
    
    # Use sample if selected
    if 'job_description' in st.session_state:
        job_description = st.session_state['job_description']
        del st.session_state['job_description']
    
    # Search functionality
    if search_button and job_description.strip():
        with st.spinner("🔍 Analyzing job description and finding matches..."):
            results = find_matching_nocs(job_description, model, embeddings, duty_embeddings, metadata, top_k)
        
        st.markdown("---")
        st.header(f"🎯 Top {len(results)} Matching NOC Codes")
        
        # Display results
        for i, result in enumerate(results, 1):
            score_percent = result['similarity_score'] * 100
            
            # Determine score color
            if score_percent >= 70:
                score_color = "#28a745"  # Green
            elif score_percent >= 50:
                score_color = "#ffc107"  # Yellow
            else:
                score_color = "#dc3545"  # Red
            
            with st.container():
                # Score breakdown
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div class="match-card">
                        <h3 style="margin: 0; color: #0066cc;">#{i} - NOC {result['noc_code']}</h3>
                        <h4 style="margin: 0.5rem 0; color: #333;">{result['title']}</h4>
                        <p style="color: #666; margin-bottom: 0.5rem;">{result['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    overall_pct = result['overall_score'] * 100
                    duty_pct = result['duty_match_score'] * 100
                    combined_pct = result['similarity_score'] * 100
                    
                    if combined_pct >= 70:
                        score_color = "#28a745"
                    elif combined_pct >= 50:
                        score_color = "#ffc107"
                    else:
                        score_color = "#dc3545"
                    
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="background-color: {score_color}; color: white; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem;">
                            <strong>{combined_pct:.1f}%</strong><br>
                            <small>Overall</small>
                        </div>
                        <div style="font-size: 0.8rem; color: #666;">
                            Context: {overall_pct:.0f}%<br>
                            Duties: {duty_pct:.0f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Matched Duties with Scores
                if result['matched_duties']:
                    with st.expander(f"🎯 Matched Duties ({len(result['matched_duties'])} matches)", expanded=True):
                        for match in result['matched_duties'][:5]:  # Show top 5
                            match_pct = match['score'] * 100
                            if match_pct >= 70:
                                badge_color = "#28a745"
                            elif match_pct >= 50:
                                badge_color = "#ffc107"
                            else:
                                badge_color = "#ff9800"
                            
                            # Highlight keywords first
                            highlighted_duty = highlight_matches(match['duty'], result['keywords'])
                            
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 0.75rem; border-radius: 5px; margin-bottom: 0.5rem; border-left: 3px solid {badge_color};">
                                <div style="display: flex; justify-content: space-between; align-items: start;">
                                    <div style="flex: 1;">
                                        <strong>NOC Duty:</strong> {highlighted_duty}<br>
                                        <small style="color: #666;">Matches: {match['matched_responsibility'][:100]}...</small>
                                    </div>
                                    <span style="background-color: {badge_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 3px; margin-left: 0.5rem; font-size: 0.85rem;">
                                        {match_pct:.0f}%
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # All Main Duties
                with st.expander("📋 All Main Duties"):
                    if result['main_duties'] and len(result['main_duties']) > 0:
                        for duty in result['main_duties']:
                            if duty and duty.strip():
                                highlighted_duty = highlight_matches(duty, result['keywords'])
                                st.markdown(f"• {highlighted_duty}", unsafe_allow_html=True)
                    else:
                        st.info("No detailed duties available")
                
                # Example Titles
                with st.expander("💼 View Example Job Titles (with highlights)"):
                    if result['example_titles'] and len(result['example_titles']) > 0:
                        for title in result['example_titles']:
                            if title and title.strip():
                                highlighted_title = highlight_matches(title, result['keywords'])
                                st.markdown(f"• {highlighted_title}", unsafe_allow_html=True)
                    else:
                        st.info("No example titles available")
                
                # Employment Requirements
                if result['employment_requirements'] and str(result['employment_requirements']).strip() and str(result['employment_requirements']) != 'nan':
                    with st.expander("📚 Employment Requirements (with highlights)"):
                        highlighted_req = highlight_matches(str(result['employment_requirements']), result['keywords'])
                        st.markdown(highlighted_req, unsafe_allow_html=True)
                
                st.markdown(f"🔗 [View Full NOC Profile]({result['url']})")
                st.markdown("---")
        
        # Export results
        st.header("📥 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            results_df = pd.DataFrame([
                {
                    'Rank': i,
                    'NOC Code': r['noc_code'],
                    'Title': r['title'],
                    'Match Score': f"{r['similarity_score']*100:.1f}%",
                    'Description': r['description'],
                    'URL': r['url']
                }
                for i, r in enumerate(results, 1)
            ])
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name="noc_matches.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON export
            json_data = pd.DataFrame(results).to_json(orient='records', indent=2)
            st.download_button(
                label="📊 Download as JSON",
                data=json_data,
                file_name="noc_matches.json",
                mime="application/json"
            )
    
    elif search_button:
        st.warning("⚠️ Please enter a job description first!")

if __name__ == "__main__":
    main()
