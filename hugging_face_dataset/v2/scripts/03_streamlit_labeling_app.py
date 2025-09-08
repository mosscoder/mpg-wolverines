#!/usr/bin/env python3
"""
Script 03: Streamlit Pelage Quality Labeling App
Interactive app for labeling pelage quality using K-means clustering visualization.
"""

import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import os
from datetime import datetime
from sklearn.cluster import KMeans
import plotly.express as px
from pathlib import Path


# Page config
st.set_page_config(
    page_title="Pelage Quality Labeling",
    page_icon="🐺",
    layout="wide"
)

# Initialize session state
if 'current_group_idx' not in st.session_state:
    st.session_state.current_group_idx = 0
if 'selected_images' not in st.session_state:
    st.session_state.selected_images = set()
if 'cluster_assignments' not in st.session_state:
    st.session_state.cluster_assignments = {}
if 'k_value' not in st.session_state:
    st.session_state.k_value = 2


class PelageLabelingApp:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        
    def connect_db(self):
        """Connect to database"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self.conn
    
    def get_unlabeled_groups(self):
        """Get unique ID/color/date groups with unlabeled images"""
        conn = self.connect_db()
        
        query = '''
        SELECT id, color, CAST(ymdh AS TEXT) as ymdh, COUNT(*) as total_images,
               SUM(CASE WHEN label IS NULL THEN 1 ELSE 0 END) as unlabeled_count
        FROM pelage_labels 
        WHERE embedding IS NOT NULL
        GROUP BY id, color, ymdh
        HAVING unlabeled_count > 0
        ORDER BY id, color, ymdh
        '''
        
        df = pd.read_sql_query(query, conn)
        return df
    
    def get_labeled_groups(self):
        """Get unique ID/color/date groups with labeled images"""
        conn = self.connect_db()
        
        query = '''
        SELECT id, color, CAST(ymdh AS TEXT) as ymdh, COUNT(*) as total_images,
               SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) as labeled_count,
               SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) as none_count,
               SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) as partial_count,
               SUM(CASE WHEN label = 2 THEN 1 ELSE 0 END) as full_count
        FROM pelage_labels 
        WHERE embedding IS NOT NULL
        GROUP BY id, color, ymdh
        HAVING labeled_count > 0
        ORDER BY id, color, ymdh
        '''
        
        df = pd.read_sql_query(query, conn)
        return df
    
    def get_group_images(self, id_val, color_val, ymdh_val, labeled_only=False, label_filter=None):
        """Get images for a specific group"""
        conn = self.connect_db()
        
        # Ensure proper types for SQLite
        ymdh_val = str(ymdh_val)
        color_val = int(color_val)  # Convert numpy int64 to Python int
        
        where_clause = "WHERE id = ? AND color = ? AND ymdh = ? AND embedding IS NOT NULL"
        params = [id_val, color_val, ymdh_val]
        
        if labeled_only:
            where_clause += " AND label IS NOT NULL"
            # Add label filter if specified
            if label_filter is not None:
                where_clause += " AND label = ?"
                params.append(int(label_filter))
        else:
            where_clause += " AND label IS NULL"
        
        query = f'''
        SELECT image_path, crop_filename, id, color, ymdh, embedding, label, labeled_at, cluster_k, cluster_id
        FROM pelage_labels 
        {where_clause}
        ORDER BY crop_filename
        '''
        
        df = pd.read_sql_query(query, conn, params=params)
        return df
    
    def get_embeddings_matrix(self, df):
        """Extract embeddings matrix from dataframe"""
        embeddings = []
        for _, row in df.iterrows():
            embedding_blob = row['embedding']
            embedding = pickle.loads(embedding_blob)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def perform_clustering(self, embeddings, k):
        """Perform K-means clustering on embeddings"""
        if len(embeddings) < k:
            k = len(embeddings)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        return cluster_labels, kmeans
    
    def save_labels(self, image_paths, label_value, k_value, cluster_assignments):
        """Save labels to database"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        current_time = datetime.now().isoformat()
        
        for image_path in image_paths:
            cluster_id = cluster_assignments.get(image_path, None)
            # Ensure proper types
            cluster_id = int(cluster_id) if cluster_id is not None else None
            cursor.execute('''
                UPDATE pelage_labels 
                SET label = ?, labeled_at = ?, cluster_k = ?, cluster_id = ?
                WHERE image_path = ?
            ''', (int(label_value), current_time, int(k_value), cluster_id, image_path))
        
        conn.commit()
        return len(image_paths)
    
    def remove_labels(self, image_paths):
        """Remove labels from images, returning them to unlabeled pool"""
        conn = self.connect_db()
        cursor = conn.cursor()
        
        for image_path in image_paths:
            cursor.execute('''
                UPDATE pelage_labels 
                SET label = NULL, labeled_at = NULL, cluster_k = NULL, cluster_id = NULL
                WHERE image_path = ?
            ''', (image_path,))
        
        conn.commit()
        return len(image_paths)
    
    def get_progress_stats(self):
        """Get overall progress statistics"""
        conn = self.connect_db()
        
        query = '''
        SELECT 
            COUNT(*) as total_images,
            SUM(CASE WHEN label IS NOT NULL THEN 1 ELSE 0 END) as labeled_images,
            COUNT(DISTINCT id || '_' || color || '_' || ymdh) as total_groups,
            COUNT(DISTINCT CASE WHEN label IS NULL THEN id || '_' || color || '_' || ymdh END) as unlabeled_groups
        FROM pelage_labels 
        WHERE embedding IS NOT NULL
        '''
        
        result = conn.execute(query).fetchone()
        return {
            'total_images': result[0],
            'labeled_images': result[1],
            'total_groups': result[2], 
            'unlabeled_groups': result[3]
        }


def display_image_grid_by_clusters(df, embeddings, cluster_labels, k):
    """Display images organized by clusters in rows"""
    
    # Group images by cluster
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    st.session_state.cluster_assignments = {
        row['image_path']: row['cluster'] 
        for _, row in df_with_clusters.iterrows()
    }
    
    # Display each cluster as a row
    for cluster_id in range(k):
        cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        if len(cluster_df) == 0:
            continue
            
        # Cluster header
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.subheader(f"Cluster {cluster_id + 1} ({len(cluster_df)} images)")
        with col2:
            if st.button(f"Select All", key=f"select_all_{cluster_id}"):
                for _, row in cluster_df.iterrows():
                    st.session_state.selected_images.add(row['image_path'])
                st.rerun()
        with col3:
            if st.button(f"Deselect All", key=f"deselect_all_{cluster_id}"):
                for _, row in cluster_df.iterrows():
                    st.session_state.selected_images.discard(row['image_path'])
                st.rerun()
        
        # Display images in columns
        images_per_row = 6
        rows_needed = (len(cluster_df) + images_per_row - 1) // images_per_row
        
        for row_idx in range(rows_needed):
            cols = st.columns(images_per_row)
            start_idx = row_idx * images_per_row
            end_idx = min(start_idx + images_per_row, len(cluster_df))
            
            for col_idx, (_, row) in enumerate(cluster_df.iloc[start_idx:end_idx].iterrows()):
                with cols[col_idx]:
                    # Load and display image
                    try:
                        img = Image.open(row['image_path'])
                        
                        # Create checkbox for selection
                        is_selected = row['image_path'] in st.session_state.selected_images
                        selected = st.checkbox(
                            f"", 
                            value=is_selected,
                            key=f"select_{row['image_path']}"
                        )
                        
                        # Handle checkbox state changes with immediate rerun for responsiveness
                        if selected != is_selected:
                            if selected:
                                st.session_state.selected_images.add(row['image_path'])
                            else:
                                st.session_state.selected_images.discard(row['image_path'])
                            st.rerun()
                        
                        # Display image with cluster color border
                        border_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                        border_color = border_colors[cluster_id % len(border_colors)]
                        
                        st.image(
                            img, 
                            width=150,
                            caption=f"{os.path.basename(row['image_path'])}",
                            use_container_width=False
                        )
                        
                        # Show current label if exists
                        if pd.notna(row.get('label')):
                            label_value = int(row['label'])
                            label_texts = {0: "None", 1: "Partial", 2: "Full"}
                            label_colors = {0: "error", 1: "warning", 2: "success"}
                            
                            label_text = label_texts.get(label_value, f"Label {label_value}")
                            label_color = label_colors.get(label_value, "info")
                            
                            if label_color == "error":
                                st.error(f"Labeled: {label_text}")
                            elif label_color == "warning":
                                st.warning(f"Labeled: {label_text}")
                            else:
                                st.success(f"Labeled: {label_text}")
                        
                    except Exception as e:
                        st.error(f"Error loading image: {e}")
        
        st.divider()


def main():
    st.title("🐺 Pelage Quality Labeling")
    
    # Database path
    db_path = "hugging_face_dataset/v2/data/labeling/pelage_labels.db"
    
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}")
        st.info("Please run 02_prepare_labeling_data.py first to generate embeddings")
        return
    
    app = PelageLabelingApp(db_path)
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        # K value slider
        k_value = st.slider("Number of Clusters (K)", min_value=2, max_value=100, value=st.session_state.k_value)
        if k_value != st.session_state.k_value:
            st.session_state.k_value = k_value
        
        # Recluster button
        if st.button("🔄 Recluster", use_container_width=True):
            st.session_state.cluster_assignments = {}
            st.rerun()
        
        # Progress stats
        st.header("Progress")
        try:
            stats = app.get_progress_stats()
            
            if stats['total_images'] > 0:
                progress_pct = (stats['labeled_images'] / stats['total_images']) * 100
                st.metric("Overall Progress", f"{progress_pct:.1f}%", 
                         f"{stats['labeled_images']}/{stats['total_images']} images")
                
                st.metric("Groups Remaining", stats['unlabeled_groups'], 
                         f"of {stats['total_groups']} total")
            
        except Exception as e:
            st.error(f"Error loading progress: {e}")
        
        # Navigation
        st.header("Navigation")
        
        # Review mode toggle
        st.divider()
        st.header("Mode")
        review_mode = st.toggle("Review Mode", help="Toggle between labeling new images and reviewing past labels")
        
        # Review mode filters
        if review_mode:
            st.divider()
            st.header("Review Filters")
            
            label_filter = st.selectbox(
                "Filter by Label",
                options=["All", "None (0)", "Partial (1)", "Full (2)"],
                help="Filter to show only specific label types"
            )
            
            # Map filter selection to database values
            filter_mapping = {
                "All": None,
                "None (0)": 0,
                "Partial (1)": 1,
                "Full (2)": 2
            }
            selected_label_filter = filter_mapping[label_filter]
        else:
            selected_label_filter = None
        
        # Labeling/Review controls in sidebar
        st.divider()
        header_text = "Review & Relabel" if review_mode else "Labeling"
        st.header(header_text)
        
        selected_count = len(st.session_state.selected_images)
        st.metric("Selected Images", selected_count)
        
        # Always show buttons but disable when no selection
        no_selection = selected_count == 0
        
        if st.button("🔴 None (0) - No pelage visible", 
                    use_container_width=True, 
                    key="none_pelage_sidebar",
                    disabled=no_selection):
            st.session_state.label_action = ('none', 0)
            st.rerun()
        
        if st.button("🟡 Partial (1) - Some pelage visible", 
                    use_container_width=True, 
                    key="partial_pelage_sidebar",
                    disabled=no_selection):
            st.session_state.label_action = ('partial', 1)
            st.rerun()
        
        if st.button("🟢 Full (2) - Full pelage visible", 
                    use_container_width=True, 
                    key="full_pelage_sidebar",
                    disabled=no_selection):
            st.session_state.label_action = ('full', 2)
            st.rerun()
        
        if st.button("🗑️ Clear Selection", 
                    use_container_width=True, 
                    key="clear_selection_sidebar",
                    disabled=no_selection):
            st.session_state.selected_images = set()
            st.rerun()
        
        # Review mode specific controls
        if review_mode and selected_count > 0:
            st.divider()
            if st.button("🔄 Remove Labels", 
                        use_container_width=True, 
                        key="remove_labels_sidebar",
                        help="Remove labels and return images to unlabeled pool"):
                st.session_state.label_action = ('remove', None)
                st.rerun()
        
        if no_selection:
            mode_text = "reviewing" if review_mode else "labeling"
            st.info(f"Select images for {mode_text}")
        
    # Get appropriate groups based on mode
    try:
        if review_mode:
            groups_df = app.get_labeled_groups()
            if len(groups_df) == 0:
                st.warning("No labeled groups found. Switch to labeling mode to create labels first.")
                return
        else:
            groups_df = app.get_unlabeled_groups()
            if len(groups_df) == 0:
                st.success("🎉 All groups have been labeled!")
                st.balloons()
                return
        
        # Group selection
        if st.session_state.current_group_idx >= len(groups_df):
            st.session_state.current_group_idx = 0
        
        current_group = groups_df.iloc[st.session_state.current_group_idx]
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.current_group_idx == 0):
                st.session_state.current_group_idx -= 1
                st.session_state.selected_images = set()
                st.rerun()
        
        with col2:
            st.info(f"Group {st.session_state.current_group_idx + 1} of {len(groups_df)}")
        
        with col3:
            if st.button("Next ➡️", disabled=st.session_state.current_group_idx >= len(groups_df) - 1):
                st.session_state.current_group_idx += 1
                st.session_state.selected_images = set()
                st.rerun()
        
        # Current group info
        st.subheader(f"Current Group: {current_group['id']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            color_text = "Color" if current_group['color'] == 1 else "B&W"
            st.metric("Image Type", color_text)
        with col2:
            st.metric("Date", current_group['ymdh'])
        with col3:
            if review_mode:
                st.metric("Labeled Images", f"{current_group['labeled_count']}/{current_group['total_images']}")
            else:
                st.metric("Unlabeled Images", f"{current_group['unlabeled_count']}/{current_group['total_images']}")
        
        # Show label distribution for review mode
        if review_mode:
            st.subheader("Label Distribution")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 None", current_group['none_count'])
            with col2:
                st.metric("🟡 Partial", current_group['partial_count'])
            with col3:
                st.metric("🟢 Full", current_group['full_count'])
        
        # Load group images
        group_images_df = app.get_group_images(
            current_group['id'], 
            current_group['color'], 
            current_group['ymdh'],
            labeled_only=review_mode,
            label_filter=selected_label_filter if review_mode else None
        )
        
        if len(group_images_df) == 0:
            if review_mode:
                filter_text = f" with filter '{label_filter}'" if label_filter != "All" else ""
                st.warning(f"No labeled images in this group{filter_text}")
            else:
                st.warning("No unlabeled images in this group")
            return
        
        # Generate embeddings matrix
        embeddings = app.get_embeddings_matrix(group_images_df)
        
        # Perform clustering
        k = min(st.session_state.k_value, len(group_images_df))
        cluster_labels, kmeans_model = app.perform_clustering(embeddings, k)
        
        # Display images by cluster
        mode_text = "Review" if review_mode else "Label"
        st.header(f"{mode_text}: Images Clustered into {k} Groups")
        display_image_grid_by_clusters(group_images_df, embeddings, cluster_labels, k)
        
        # Handle labeling/review actions from sidebar
        if hasattr(st.session_state, 'label_action') and st.session_state.label_action:
            action_type, label_value = st.session_state.label_action
            if len(st.session_state.selected_images) > 0:
                if action_type == 'remove':
                    # Remove labels
                    removed_count = app.remove_labels(list(st.session_state.selected_images))
                    st.success(f"✅ Removed labels from {removed_count} images")
                else:
                    # Apply or change labels
                    saved_count = app.save_labels(
                        list(st.session_state.selected_images), 
                        label_value, 
                        k, 
                        st.session_state.cluster_assignments
                    )
                    # Map label values to descriptive text
                    label_texts = {0: "None (no pelage visible)", 1: "Partial (some pelage visible)", 2: "Full (full pelage visible)"}
                    label_text = label_texts.get(label_value, f"Label {label_value}")
                    action_verb = "Relabeled" if review_mode else "Labeled"
                    st.success(f"✅ {action_verb} {saved_count} images as {label_text}")
                
                st.session_state.selected_images = set()
                st.session_state.label_action = None
                st.rerun()
            else:
                action_text = "reviewing" if review_mode else "labeling"
                st.warning(f"No images selected for {action_text}")
                st.session_state.label_action = None
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()