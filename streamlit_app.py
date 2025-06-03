import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import ast
import re

# Data Loading
@st.cache_data(ttl=3600)
def load_data():
    try:
        deepface_df = pd.read_csv("deepface_emotions_combined.csv")
        test_df = pd.read_csv("test_set_labels.csv")
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    merged = test_df.merge(
        deepface_df[['Image', 'Dominant Emotion', 'Face Confidence', 'url']],
        left_on='Image ID',
        right_on='Image',
        how='left'
    ).drop(columns=['Image'])
    merged.rename(columns={'Dominant Emotion': 'Predicted Emotion'}, inplace=True)
    return merged, deepface_df

# Sidebar Controls

data_scope = st.sidebar.radio(
    "Select Data Scope",
    options=["Test Set Only", "All Images"],
    index=0
)
view = st.sidebar.radio(
    "Select View",
    options=[
        "Overview of the data",
        "Model accuracy",
        "Interactive sankey diagram",
        "Emotions visualized over time",
        "Stacked bar chart"
    ],
    index=0
)

merged, deepface_df = load_data()
min_conf = st.sidebar.slider("Minimum Face Confidence", min_value=0.85, max_value=1.0, value=0.85, step=0.01)

if 'Face Confidence' in merged.columns:
    merged = merged[merged['Face Confidence'] >= min_conf]
if 'Face Confidence' in deepface_df.columns:
    deepface_df = deepface_df[deepface_df['Face Confidence'] >= min_conf]

if data_scope == "Test Set Only":
    df = merged.copy()
else:
    df = deepface_df.copy()
    df = df.drop_duplicates()
    df = df.rename(columns={
        'Image': 'Image ID',
        'Folder': 'Source',
        'Person': 'Politician',
        'Dominant Emotion': 'Predicted Emotion'
    })
    df = df.drop_duplicates(subset=["Image ID", "Politician"])
    df['Predicted Emotion'] = df['Predicted Emotion'].astype(str).str.strip().str.lower()

    # ✂ Insert this block to simplify your `Source` column ✂
    def simplify_source(raw):
        """
        - If the folder path contains "instagram_posts", return "Instagram".
        - If it contains "svc_classification_corrected", extract the next part
          (typically "NOS_t1" or "NU_t1") and return that.
        - Otherwise, leave it as-is.
        """
        s = str(raw)
        s_low = s.lower()
        if 'instagram_posts' in s_low:
            return "Instagram"
        if 'svc_classification_corrected' in s_low:
            parts = s.split('/')
            try:
                idx = parts.index('svc_classification_corrected')
                # The subfolder right after "svc_classification_corrected" is usually "NOS_t1" or "NU_t1"
                if idx + 1 < len(parts):
                    return parts[idx + 1]
            except ValueError:
                pass
        return s  # fallback to the original string if neither case matches

    df['Source'] = df['Source'].apply(simplify_source)

orientation_data = pd.read_csv("Political_Orientation_Data.csv")
df = df.merge(orientation_data, how='left', left_on='Politician', right_on='Person')

if "Left/Right" not in df.columns or "Progressive/Conservative" not in df.columns:
    st.error("The columns 'Left/Right' and 'Progressive/Conservative' are missing from the dataset.")

df['Left/Right Category'] = df['Left/Right'].apply(lambda x: 'Left' if x < 0 else 'Right')
df['Progressive/Conservative Category'] = df['Progressive/Conservative'].apply(lambda x: 'Progressive' if x > 0 else 'Conservative')

seat_data = {
    "Caroline van der Plas": 1,
    "Dilan Yesilgoz": 34,
    "Edson olf": 1,
    "Frans Timmermans": 17,
    "Geert Wilders": 17,
    "Henri Bontebal": 15,
    "Joost Eerdmans": 3,
    "Kees van der Staaij": 3,   
    "Laurens Dassen": 3,
    "Lilian Marijnissen": 9,
    "Mirjam Bikker": 6,
    "Pieter Omtzigt": 1,
    "Rob Jetten": 24,
    "Stephan van Baarle": 3,
    "Thierry Baudet": 8,
    "Wybren van Haga": 1,
}
df['Seats'] = df['Politician'].map(seat_data).fillna(0)

total_seats = df['Seats'].sum() if 'Seats' in df.columns else 0
df['Seat Weight'] = df['Seats'] / total_seats if total_seats > 0 else 1

if 'Photo Count' not in df.columns:
    df['Photo Count'] = df.groupby('Politician')['Image ID'].transform('count')
df['Photo Count'] = pd.to_numeric(df['Photo Count'], errors='coerce').fillna(1)
photo_counts = df.groupby('Politician')['Photo Count'].first()
photo_weights = photo_counts ** -0.5
photo_weights = photo_weights / photo_weights.sum() if photo_weights.sum() > 0 else 1

df['Photo Weight'] = df['Politician'].map(photo_weights)
if df['Photo Weight'].sum() == 0:
    df['Photo Weight'] = 1 / len(df)

st.sidebar.markdown("### Apply Weights")
apply_seat_weight = st.sidebar.checkbox("Seat-based Weight")
apply_photo_weight = st.sidebar.checkbox("Photo-based Weight")

if apply_seat_weight and apply_photo_weight:
    df['Combined Weight'] = df['Seat Weight'] * df['Photo Weight']
elif apply_seat_weight:
    df['Combined Weight'] = df['Seat Weight']
elif apply_photo_weight:
    df['Combined Weight'] = df['Photo Weight']
else:
    df['Combined Weight'] = 1

with st.sidebar.expander('Weights per Politician', expanded=True):
    st.dataframe(
        pd.DataFrame({
            'Seat Weight': df.groupby('Politician')['Seat Weight'].first(),
            'Photo Count': photo_counts,
            'Photo Weight': photo_weights
        }),
        use_container_width=True
    )

def parse_category(cell):
    if pd.isna(cell) or not isinstance(cell, str):
        return []
    cats = re.findall(r"'([^']+)'", cell)
    if cats:
        return cats
    return [x.strip() for x in re.split(r"[,;]\s*", cell) if x.strip()]

if 'category' in df.columns:
    df['category'] = df['category'].apply(parse_category)
    df = df.explode('category')

st.title("Emotion Analysis Dashboard")

# --- Ensure Date column is present and correct for Instagram posts ---
if 'Date' not in df.columns:
    df['Date'] = pd.NaT

# For Instagram: if Date is missing, extract from Image ID (filename)
mask_instagram = (df['Source'].str.lower().isin(['instagram_posts', 'instagram']))
missing_date = mask_instagram & (df['Date'].isna() | (df['Date'] == ''))

def extract_date_from_filename(filename):
    # Accepts formats like 'YYYYMMDD' or 'YYMMDD' at the start of the filename
    import re
    match = re.match(r'(\d{8})', str(filename))
    if match:
        return pd.to_datetime(match.group(1), format='%Y%m%d', errors='coerce')
    match = re.match(r'(\d{6})', str(filename))
    if match:
        return pd.to_datetime(match.group(1), format='%d%m%y', errors='coerce')
    return pd.NaT

if 'Image ID' in df.columns:
    df.loc[missing_date, 'Date'] = df.loc[missing_date, 'Image ID'].apply(extract_date_from_filename)

if view == "Overview of the data":
    st.subheader("Welcome to the Emotion Analysis Dashboard")
    st.markdown("### About This Dashboard")
    st.info(
        """
        This dashboard provides an interactive way to explore emotion analysis data.\n\n
        - **Photo Collage**: A random selection of images from the test set.  
        - **Data Overview**: A preview of all images (news & Instagram).  
        - **Interactive Visualizations**: Explore confusion matrices, Sankey diagrams, radar charts, and more.
        """
    )

    total_images = len(df)
    st.write(f"Total number of images: {total_images}")

    # 1) Instead of using raw df['Source'], create a simple Source_Display column:
    #    Anything containing "instagram" → "Instagram"; else → "News"
    def map_to_display(src):
        s = str(src).lower()
        if 'instagram' in s:
            return "Instagram"
        else:
            return "News"

    df['Source_Display'] = df['Source'].apply(map_to_display)

    # 2) Build our filter widgets using Source_Display (so "Instagram" and "News" show up)
    all_politicians = sorted(df['Politician'].dropna().unique())
    all_emotions   = sorted(df['Predicted Emotion'].dropna().unique())
    all_sources    = sorted(df['Source_Display'].dropna().unique())  # now just ["Instagram","News"]

    col1, col2 = st.columns(2)
    with col1:
        selected_person  = st.selectbox("Filter by Politician", ["All"] + all_politicians)
    with col2:
        selected_emotion = st.selectbox("Filter by Predicted Emotion", ["All"] + all_emotions)

    # 3) This multiselect now shows only "Instagram" and "News"
    selected_sources = st.multiselect("Filter by Source", all_sources, default=all_sources)

    # 4) Filter df by Source_Display (instead of raw Source)
    filtered_df = df[df['Source_Display'].isin(selected_sources)]

    if selected_person  != "All":
        filtered_df = filtered_df[filtered_df['Politician'] == selected_person]
    if selected_emotion != "All":
        filtered_df = filtered_df[filtered_df['Predicted Emotion'] == selected_emotion]

    filtered_df = filtered_df.drop_duplicates(subset=["Image ID"])
    st.markdown(f"**{len(filtered_df)} images found**")

    # 5) Show the table of raw columns (including the real ‘Source’, so you can inspect the folder path if you want)
    display_cols = ['Image ID', 'Politician', 'Source', 'Predicted Emotion', 'Face Confidence', 'url']
    if 'Actual Emotion' in filtered_df.columns:
        display_cols.insert(3, 'Actual Emotion')
    if 'Date' in filtered_df.columns and 'Date' not in display_cols:
        display_cols.append('Date')

    if not filtered_df.empty:
        st.dataframe(filtered_df[display_cols].sort_values('Face Confidence', ascending=False))
    else:
        st.warning("No images found for the selected filters.")

    # 6) Build the collage exactly as before, but now only over 'filtered_df'
    if not filtered_df.empty:
        rows = filtered_df.iterrows()
        for i, row_group in enumerate(zip(*[rows] * 3)):
            cols = st.columns(3)
            for col, (_, row) in zip(cols, row_group):
                source        = row['Source']            # raw folder, e.g. "Instagram" or "NOS_t1"
                src_display   = row['Source_Display']    # "Instagram" or "News"
                image_name    = row['Image ID']
                person        = row['Politician']
                actual        = row.get('Actual Emotion', '?')
                predicted     = row.get('Predicted Emotion', '?')
                link          = row.get('url', '?')

                if data_scope == "Test Set Only":
                    image_path = f"images/test_set/{source}/{person}/{image_name}"
                else:  # data_scope == "All Images"
                    if src_display == "Instagram":
                        # ← MAKE SURE this matches your actual folder (capitalization matters!)
                        image_path = f"Images/Instagram_posts/{person}/{image_name}"
                    else:  # News
                        # e.g. source might be "NOS_t1" or "NU_t1"
                        subfolder = source  # because raw Source is exactly "NOS_t1" or "NU_t1"
                        image_path = f"Images/svc_classification_corrected/{subfolder}/{person}/{image_name}"

                source_info = f"Source: {source}"
                caption = (
                    f"Photo ID: {image_name} | {person} | Actual: {actual} | "
                    f"Predicted: {predicted} | {source_info} | link: {link}"
                )

                try:
                    col.image(image_path, caption=caption, use_container_width=True)
                except FileNotFoundError:
                    col.warning(f"Could not load: {image_path}")
    else:
        st.warning("No images to display in collage.")

elif view == "Model accuracy":
    st.subheader("Confusion Matrix and Prediction Statistics")
    st.markdown(
        """
        The model used for emotion analysis classification is **DeepFace** and runs on all photos seen on the main page.\n\nThe predictions are manually verified for 'Actual' emotion and have (if available per person) **15 images per source per person**, which provides a somewhat clear view of the accuracy of the model. The prediction heat map and overall accuracy, precision, recall, and F1-score can be seen below.
        """
    )
    if 'Actual Emotion' in merged.columns and 'Predicted Emotion' in merged.columns:
        merged['Actual Emotion'] = merged['Actual Emotion'].str.strip().str.lower()
        merged['Predicted Emotion'] = merged['Predicted Emotion'].str.strip().str.lower()
        valid_df = merged.dropna(subset=['Actual Emotion', 'Predicted Emotion'])
    else:
        valid_df = pd.DataFrame()
    if valid_df.empty:
        st.warning("No valid data available for the confusion matrix.")
    else:
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
        import plotly.figure_factory as ff
        actual = valid_df['Actual Emotion']
        predicted = valid_df['Predicted Emotion']
        labels = sorted(set(actual.unique()).union(set(predicted.unique())))
        cm = confusion_matrix(actual, predicted, labels=labels)
        accuracy = accuracy_score(actual, predicted) * 100
        precision = precision_score(actual, predicted, average='weighted', zero_division=0) * 100
        recall = recall_score(actual, predicted, average='weighted', zero_division=0) * 100
        f1 = f1_score(actual, predicted, average='weighted', zero_division=0) * 100
        st.markdown("### Overall Metrics")
        st.write(f"**Accuracy:** {accuracy:.2f}%")
        st.write(f"**Precision (Weighted):** {precision:.2f}%")
        st.write(f"**Recall (Weighted):** {recall:.2f}%")
        st.write(f"**F1-Score (Weighted):** {f1:.2f}%")
        st.markdown("### Confusion Matrix")
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Viridis",
            showscale=True,
            annotation_text=cm.astype(str),
            hoverinfo="z"
        )
        fig.update_layout(
            title="Confusion Matrix Heatmap",
            xaxis=dict(title="Predicted Emotion", tickangle=-45, side="bottom"),
            yaxis=dict(title="Actual Emotion"),
            height=600,
            width=600
        )
        st.plotly_chart(fig)
        st.markdown("### Classification Report")
        report = classification_report(actual, predicted, labels=labels, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        if 'accuracy' in report_df.index:
            styled_report_df = report_df.drop(index=['accuracy']).style.format("{:.2f}").background_gradient(cmap="Blues")
        else:
            styled_report_df = report_df.style.format("{:.2f}").background_gradient(cmap="Blues")
        st.dataframe(styled_report_df)
        if 'Face Confidence' in merged.columns:
            bins = np.arange(0.88, 1.01, 0.01)
            merged['Correct'] = merged['Actual Emotion'].str.lower() == merged['Predicted Emotion'].str.lower()
            merged['conf_bin'] = pd.cut(merged['Face Confidence'], bins=bins, include_lowest=True, right=False)
            merged['conf_bin_mid'] = merged['conf_bin'].apply(lambda x: x.left + 0.005 if pd.notnull(x) else np.nan)
            bin_edges = bins[:-1]
            cum_stats = []
            for edge in bin_edges:
                subset = merged[merged['Face Confidence'] >= edge]
                total = len(subset)
                correct = subset['Correct'].sum()
                incorrect = total - correct
                accuracy = correct / total if total > 0 else np.nan
                cum_stats.append({
                    'conf_bin_mid': edge + 0.005,
                    'total': total,
                    'correct': correct,
                    'incorrect': incorrect,
                    'accuracy': accuracy
                })
            stats = pd.DataFrame(cum_stats)
            all_emotions = sorted(valid_df['Actual Emotion'].unique())
            incorrect_df = merged[~merged['Correct']]
            incorrect_counts = incorrect_df['Actual Emotion'].value_counts().reindex(all_emotions, fill_value=0)
            total_counts = merged.groupby('Actual Emotion').size().reindex(all_emotions, fill_value=0).astype(int)
            correct_counts = merged[merged['Correct']].groupby('Actual Emotion').size().reindex(all_emotions, fill_value=0).astype(int)
            percent_correct = (correct_counts / total_counts * 100).round(2).replace([np.inf, -np.inf, np.nan], 0)
            summary_df = pd.DataFrame({
                "Incorrect Predictions": incorrect_counts,
                f"Total Predictions (≥ {min_conf:.2f})": total_counts,
                "Correct Prediction %": percent_correct
            })
            st.markdown("### Incorrect Predictions per Actual Emotion")
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True, height=min(500, 40 + 35 * len(summary_df)))
            else:
                st.info("No incorrect predictions in the current selection.")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=stats['conf_bin_mid'],
                y=stats['correct'],
                mode='lines+markers',
                name='Cumulative Correct Predictions',
                line=dict(color='green')
            ))
            fig2.add_trace(go.Scatter(
                x=stats['conf_bin_mid'],
                y=stats['incorrect'],
                mode='lines+markers',
                name='Cumulative Incorrect Predictions',
                line=dict(color='red')
            ))
            fig2.add_trace(go.Scatter(
                x=stats['conf_bin_mid'],
                y=stats['accuracy'],
                mode='lines+markers',
                name='Cumulative Average Prediction Rate',
                yaxis='y2',
                line=dict(color='blue', dash='dash')
            ))
            fig2.update_layout(
                xaxis_title="Minimum Face Confidence",
                yaxis=dict(
                    title="Cumulative Count (Correct/Incorrect)",
                    showgrid=True,
                    gridcolor="rgba(200, 200, 200, 0.7)",
                    gridwidth=1,
                    zeroline=True,
                    zerolinecolor="rgba(150, 150, 150, 0.8)",
                    zerolinewidth=2,
                    tickmode='auto'
                ),
                yaxis2=dict(
                    title="Cumulative Average Prediction Rate",
                    overlaying='y',
                    side='right',
                    range=[0, 1],
                    showgrid=False
                ),
                legend_title="Metric",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.08,
                    xanchor="center",
                    x=0.5
                ),
                height=400
            )
            st.markdown("### Per-Politician, Per-Emotion Counts and Accuracy")
            st.markdown("This table shows the count of each emotion per politician, along with the accuracy of predictions for that emotion.")
            emotion_list = sorted(valid_df['Actual Emotion'].unique())
            politician_list = sorted(valid_df['Politician'].unique())
            count_df = valid_df.groupby(['Politician', 'Actual Emotion']).size().unstack(fill_value=0).reindex(index=politician_list, columns=emotion_list, fill_value=0)
            correct_mask = valid_df['Actual Emotion'] == valid_df['Predicted Emotion']
            correct_df = valid_df[correct_mask].groupby(['Politician', 'Actual Emotion']).size().unstack(fill_value=0).reindex(index=politician_list, columns=emotion_list, fill_value=0)
            table_data = {}
            for pol in politician_list:
                row = {}
                for emo in emotion_list:
                    total = count_df.loc[pol, emo]
                    correct = correct_df.loc[pol, emo]
                    acc = (correct / total * 100) if total > 0 else 0
                    row[emo] = f"{total} ({acc:.0f}%)" if total > 0 else ""
                table_data[pol] = row
            summary_pol_emotion = pd.DataFrame.from_dict(table_data, orient='index', columns=emotion_list)
            st.dataframe(summary_pol_emotion, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True, key="face_confidence_plot")
        else:
            st.warning("Required columns 'Actual Emotion' and 'Predicted Emotion' not found in the dataset.")

elif view == "Interactive sankey diagram":
    st.markdown(
        """
        In this Sankey diagram, you can select the source, left/right-wing and conservative/progressive parties, predicted emotion, and actual emotion. You can then select a party leader to track their contribution to the Sankey diagram and add a seat-based weight to better understand the effect of certain politicians on the emotional weight and representation in the news and within parties. For better understandability, you can move the slider of the whitespace to expand or shorten distances between lines and blocks.
        """
    )
    st.subheader("Sankey Diagram: Multi-Level Emotion Flow with Highlighting")
    col1, col2 = st.columns([1, 2])
    with col1:
        whitespace = st.slider("Adjust Whitespace", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    with col2:
        available_levels = ["Source", "Predicted Emotion"]
        if data_scope == "Test Set Only":
            available_levels.append("Actual Emotion")
        if "Left/Right Category" in df.columns:
            available_levels.append("Left/Right Category")
        if "Progressive/Conservative Category" in df.columns:
            available_levels.append("Progressive/Conservative Category")
        selected_levels = st.multiselect(
            "Select Levels for Sankey Diagram (Order Matters)",
            options=available_levels,
            default=["Source", "Predicted Emotion"]
        )
    all_politicians = sorted(df['Politician'].dropna().unique())
    selected_politicians = st.multiselect("Highlight/Select Politicians", all_politicians, default=[])
    custom_politician_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#f781bf", "#a65628", "#999999", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a6cee3"
    ]
    color_map = {politician: custom_politician_colors[i % len(custom_politician_colors)] for i, politician in enumerate(all_politicians)}
    if len(selected_levels) < 2:
        st.warning("Please select at least two levels for the Sankey diagram.")
    else:
        try:
            sankey_data = df.groupby(selected_levels + ['Politician'])['Combined Weight'].sum().reset_index(name="Weighted Count")
            for level in selected_levels:
                total_weight = sankey_data.groupby(level)['Weighted Count'].sum()
                small_values = total_weight[total_weight / total_weight.sum() < 0.05].index
                sankey_data[level] = sankey_data[level].apply(lambda x: "Other" if x in small_values else x)
            if selected_politicians:
                sankey_data['highlight_order'] = sankey_data['Politician'].apply(
                    lambda x: selected_politicians.index(x) if x in selected_politicians else len(selected_politicians)
                )
                sankey_data = sankey_data.sort_values(['highlight_order'] + selected_levels)
                sankey_data = sankey_data.drop(columns=['highlight_order'])
            else:
                sankey_data = sankey_data.sort_values(selected_levels + ['Politician'])
            node_indices = {}
            all_nodes = []
            for level in selected_levels:
                unique_nodes = []
                for node in sankey_data[level]:
                    if node not in unique_nodes:
                        unique_nodes.append(node)
                node_indices[level] = {node: len(all_nodes) + i for i, node in enumerate(unique_nodes)}
                all_nodes.extend(unique_nodes)
            links = {"source": [], "target": [], "value": [], "color": []}
            for _, row in sankey_data.iterrows():
                for i in range(len(selected_levels) - 1):
                    source_idx = node_indices[selected_levels[i]][row[selected_levels[i]]]
                    target_idx = node_indices[selected_levels[i + 1]][row[selected_levels[i + 1]]]
                    value = row["Weighted Count"]
                    if row['Politician'] in selected_politicians:
                        link_color = color_map[row['Politician']]
                    else:
                        link_color = "rgba(100, 100, 100, 0.5)"
                    links["source"].append(source_idx)
                    links["target"].append(target_idx)
                    links["value"].append(value)
                    links["color"].append(link_color)
            node_labels = [f"{node}" for node in all_nodes]
            fig = go.Figure(data=[go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=int(whitespace * 100),
                    thickness=40,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color="rgba(200, 200, 200, 0.8)",
                ),
                link=dict(
                    source=links["source"],
                    target=links["target"],
                    value=links["value"],
                    color=links["color"]
                )
            )])
            fig.update_layout(
                title_text="Multi-Level Sankey Diagram with Proper Alignment of 'Other' Groups",
                font=dict(family="Arial, sans-serif", size=12, color="black"),
                plot_bgcolor='rgba(240, 240, 240, 1)',
                paper_bgcolor='white',
                height=600
            )
            st.plotly_chart(fig)
            if selected_politicians:
                st.markdown("### Legend for Highlighted Politicians")
                for politician in selected_politicians:
                    st.markdown(f"<span style='color:{color_map[politician]};'>■</span> {politician}", unsafe_allow_html=True)
        except KeyError as e:
            st.error(f"Error creating Sankey diagram: {e}")

elif view == "Emotions visualized over time":
    st.subheader("Stream Graph: Emotion Trends Over Time")
    st.markdown(
        """
        Here you can see the overall posts on Instagram or news sources of all the party leaders combined or individually over time.

        You can show either absolute counts or percentages, and group the total amount of images per a time frame of days, three days, weeks, two weeks, or months.

        Use the dropdown below to select the source (e.g., "Instagram", "NOS_t1", or "NU_t1").
        """
    )

    # 1) Ensure 'Date' exists; parse any existing values (invalid → NaT)
    if 'Date' not in df.columns:
        df['Date'] = pd.NaT
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # 2) For Instagram rows with missing Date, attempt to extract from 'Image ID'
    mask_instagram = df['Source'].str.lower().str.contains('instagram')
    missing_date_instagram = mask_instagram & df['Date'].isna()

    def extract_date_from_filename(filename):
        fname = str(filename)
        # Try DDMMYY (6 digits) at start
        m = re.match(r'^(\d{6})', fname)
        if m:
            return pd.to_datetime(m.group(1), format='%d%m%y', errors='coerce')
        # Try YYYYMMDD (8 digits) at start
        m = re.match(r'^(\d{8})', fname)
        if m:
            return pd.to_datetime(m.group(1), format='%Y%m%d', errors='coerce')
        return pd.NaT

    if 'Image ID' in df.columns:
        df.loc[missing_date_instagram, 'Date'] = (
            df.loc[missing_date_instagram, 'Image ID']
              .apply(extract_date_from_filename)
        )

    # Debug: Show Instagram date coverage
    if mask_instagram.any():
        ig_dates = df.loc[mask_instagram, 'Date']
        unique_dates = ig_dates.dropna().unique()
        st.info(f"Instagram images: {len(unique_dates)} unique dates. Range: {ig_dates.min()} to {ig_dates.max()}")

    # 3) Drop any rows that still have NaT in 'Date'
    df = df.dropna(subset=['Date'])

    # 4) “Select Source” dropdown pulls directly from the raw df['Source']
    available_sources = sorted(df['Source'].dropna().unique())
    selected_source = st.selectbox("Select Source", available_sources, index=0)

    # 5) Filter to only the chosen source (Instagram, NOS_t1, NU_t1, etc.)
    filtered_df = df[df['Source'] == selected_source].copy()

    # 6) Re‐compute orientation/ideology columns (if not already present)
    if 'Left/Right' in filtered_df.columns:
        filtered_df['Orientation'] = filtered_df['Left/Right'].apply(lambda x: 'Left' if x < 0 else 'Right')
    else:
        filtered_df['Orientation'] = np.nan

    if 'Progressive/Conservative' in filtered_df.columns:
        filtered_df['Ideology'] = filtered_df['Progressive/Conservative'].apply(lambda x: 'Progressive' if x > 0 else 'Conservative')
    else:
        filtered_df['Ideology'] = np.nan

    # 7) Politician multi‐select
    all_politicians = sorted(filtered_df['Politician'].dropna().unique())
    selected_politicians = st.multiselect("Select Politicians", all_politicians, default=all_politicians)

    # 8) Group‐filter radio buttons
    group_filter = st.radio(
        "Filter by Group",
        options=["All", "Left", "Right", "Progressive", "Conservative"],
        index=0
    )

    # 9) Time bin options
    bin_freq_options = {
        "Monthly (4 bins max)": "1M",
        "Biweekly":            "14D",
        "Weekly (default)":    "7D",
        "Every 3 Days":        "3D",
        "Daily":               "1D"
    }
    bin_label = st.selectbox("Select Bin Size", list(bin_freq_options.keys()), index=2)
    freq = bin_freq_options[bin_label]

    # 10) Toggle: absolute count vs. percentage
    show_percentage = st.checkbox("Show as Percentage", value=False)

    # 11) Filter by politician(s)
    if selected_politicians:
        filtered_df = filtered_df[filtered_df['Politician'].isin(selected_politicians)]

    # 12) Filter by ideological group (if not “All”)
    if group_filter != "All":
        if group_filter in ["Left", "Right"]:
            filtered_df = filtered_df[filtered_df['Orientation'] == group_filter]
        else:  # Progressive/Conservative
            filtered_df = filtered_df[filtered_df['Ideology'] == group_filter]

    # 13) If no data remains, warn the user
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # 14) Always aggregate using the selected bin size first
        grouped = (
            filtered_df
            .set_index('Date')
            .groupby([pd.Grouper(freq=freq), 'Predicted Emotion'])
            .agg({'Combined Weight': 'sum'})
            .reset_index()
            .rename(columns={'Combined Weight': 'Count'})
        )

        # --- Performance safeguard: if too many bins, force coarser binning ---
        max_bins = 100
        n_bins = grouped['Date'].nunique()
        if n_bins > max_bins:
            # Find next coarser bin
            freq_order = ['1D', '3D', '7D', '14D', '1M']
            current_idx = freq_order.index(freq) if freq in freq_order else len(freq_order) - 1
            while n_bins > max_bins and current_idx < len(freq_order) - 1:
                current_idx += 1
                coarser_freq = freq_order[current_idx]
                grouped = (
                    filtered_df
                    .set_index('Date')
                    .groupby([pd.Grouper(freq=coarser_freq), 'Predicted Emotion'])
                    .agg({'Combined Weight': 'sum'})
                    .reset_index()
                    .rename(columns={'Combined Weight': 'Count'})
                )
                n_bins = grouped['Date'].nunique()
            st.info(f"Too many time bins for plotting ({n_bins}). Automatically switched to coarser bin size: {coarser_freq}.")
            freq = coarser_freq
            bin_label = [k for k, v in bin_freq_options.items() if v == freq][0] if freq in bin_freq_options.values() else freq

        # 15) If “Show as Percentage,” convert aggregated counts to percentages per bin
        if show_percentage:
            total_per_bin = grouped.groupby('Date')['Count'].transform('sum')
            grouped['Count'] = (grouped['Count'] / total_per_bin) * 100
            y_axis_label = "Percentage of Weighted Photos"
        else:
            y_axis_label = "Weighted Number of Photos"

        # 16) Define emotion colors
        emotion_colors = {
            'angry':    '#FF6F61',
            'disgust':  '#6B8E23',
            'fear':     '#8A2BE2',
            'happy':    '#FFD700',
            'sad':      '#1E90FF',
            'surprise': '#FF8C00',
            'neutral':  '#A9A9A9'
        }

        # 17) Create the area chart directly from the grouped DataFrame (long form)
        fig = px.area(
            grouped,
            x='Date',
            y='Count',
            color='Predicted Emotion',
            color_discrete_map=emotion_colors,
            line_group='Predicted Emotion',
            title=f"{selected_source} Emotion Streamgraph ({bin_label})",
            labels={'Date': 'Date', 'Count': y_axis_label},
            height=500
        )
        # Make the lines smooth (spline interpolation)
        fig.update_traces(mode='lines', line_shape='spline')

        # 18) Remove all x-axis ticks and labels for a clean look
        fig.update_layout(
            xaxis=dict(
                showticklabels=False,
                ticks='',
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.5)",
                title="Date"
            ),
            yaxis=dict(
                title=y_axis_label,
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.5)"
            ),
            legend=dict(
                title="Emotion",
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ),
            hovermode="x unified",
            plot_bgcolor='rgba(240, 240, 240, 1)',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="black")
        )

        # 19) Display the chart
        st.plotly_chart(fig)

elif view == "Stacked bar chart":
    st.subheader("Stacked Bar Chart: Emotional Distribution")
    st.markdown(
        """
        This chart shows the emotional distribution for selected categories, either as counts or normalized percentages. Select grouping dimension, filter which groups to include, and toggle between count vs percentage.
        """
    )
    df_expanded = df.copy()
    if 'category' in df_expanded.columns:
        def parse_cat(val):
            if pd.isna(val) or val is None:
                return []
            val_str = str(val).strip()
            cats = re.findall(r"'([^']+)'", val_str)
            if cats:
                return cats
            return [x.strip() for x in re.split(r"[,;]\s*", val_str) if x.strip()]
        df_expanded['category'] = df_expanded['category'].apply(parse_cat)
        df_expanded = df_expanded.explode('category')
    if 'Politician' not in df_expanded or 'Predicted Emotion' not in df_expanded:
        st.warning("Required columns 'Politician' and 'Predicted Emotion' not found.")
    else:
        all_sources = sorted(df_expanded['Source'].dropna().unique())
        available_groupings = []
        if 'category' in df_expanded.columns:
            available_groupings.append('category')
        for col in ['Politician', 'Left/Right Category', 'Progressive/Conservative Category', 'Source']:
            if col in df_expanded.columns:
                available_groupings.append(col)
        group_by = st.selectbox("Group by", available_groupings)
        selected_sources = st.multiselect("Filter by Source", all_sources, default=all_sources)
        df_filtered = df_expanded[df_expanded['Source'].isin(selected_sources)]
        available_groups = sorted(df_filtered[group_by].dropna().unique())
        selected_groups = st.multiselect(f"Select {group_by}s to include", available_groups, default=available_groups)
        df_filtered = df_filtered[df_filtered[group_by].isin(selected_groups)]
        show_pct = st.checkbox("Show as percentage", value=True)
        if df_filtered.empty:
            st.warning("No data available after filtering.")
        else:
            grouped = (
                df_filtered.groupby([group_by, 'Predicted Emotion'])['Combined Weight']
                .sum().reset_index()
                .pivot(index=group_by, columns='Predicted Emotion', values='Combined Weight')
                .fillna(0)
            )
            if show_pct:
                grouped = grouped.div(grouped.sum(axis=1), axis=0) * 100
                yaxis_label = "Percentage"
            else:
                yaxis_label = "Count"
            fig = go.Figure()
            for emo in grouped.columns:
                fig.add_trace(go.Bar(x=grouped.index, y=grouped[emo], name=emo))
            fig.update_layout(
                barmode='stack',
                title=f"Emotional Distribution by {group_by}",
                xaxis_title=group_by,
                yaxis_title=yaxis_label,
                yaxis=dict(tickformat=".0f"),
                legend_title="Emotion",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)


