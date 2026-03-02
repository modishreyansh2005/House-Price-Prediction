import streamlit as st
import pandas as pd
import pickle as pk

# Load trained model
@st.cache_resource
def load_model():
    try:
        with open('House_prediction_model.pkl', 'rb') as f:
            return pk.load(f)
    except FileNotFoundError:
        st.error("❌ Model file not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Load dataset
@st.cache_data
def load_data():
    try:
        return pd.read_csv('House Price Prediction Dataset.csv')
    except FileNotFoundError:
        st.error("❌ Dataset file not found!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# Main app
def main():
    st.set_page_config(page_title="House Price Prediction", page_icon="🏡")
    
    st.title('🏡 House Price Prediction')
    st.write('Enter the details below to predict the house price')
    
    # Load model and data
    model = load_model()
    data = load_data()
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        loc = st.selectbox('🗺️ Choose the location', sorted(data['Location'].unique()))
        area = st.number_input('📏 Enter total sqft', min_value=0.0, step=10.0, value=1000.0)
        beds = st.number_input('🛏️ Enter No of Bedrooms', min_value=0, max_value=20, step=1, value=2)
    
    with col2:
        bath = st.number_input('🚿 Enter No of Bathrooms', min_value=0, max_value=20, step=1, value=2)
        floor = st.number_input('🏢 Enter No of Floors', min_value=0, max_value=10, step=1, value=1)
    
    # Add some spacing
    st.write("")
    
    # Prediction button
    if st.button("🔮 Predict Price", type="primary"):
        
        if area == 0:
            st.warning("⚠️ Please enter a valid area (sqft)")
        else:
            # Create input DataFrame
            input_df = pd.DataFrame([{
                'Location': loc,
                'Area': area,
                'Bedrooms': beds,
                'Bathrooms': bath,
                'Floors': floor,
            }])
            
            try:
                # Check if model expects one-hot encoded features
                if hasattr(model, 'feature_names_in_'):
                    # Get dummy variables for Location
                    input_encoded = pd.get_dummies(input_df, columns=['Location'])
                    
                    # Get column names model was trained on
                    train_columns = model.feature_names_in_
                    
                    # Add missing columns with 0
                    for col in train_columns:
                        if col not in input_encoded.columns:
                            input_encoded[col] = 0
                    
                    # Reorder columns to match training data
                    input_encoded = input_encoded[train_columns]
                    
                    # Make prediction
                    output = model.predict(input_encoded)
                else:
                    # Direct prediction
                    output = model.predict(input_df)
                
                # Display result
                predicted_price = round(output[0], 2)
                
                st.success(f"### 🏠 Predicted House Price: ${predicted_price:,.2f}")
                
                # Show input summary
                with st.expander("📋 Input Summary"):
                    st.write(f"**Location:** {loc}")
                    st.write(f"**Area:** {area:,.0f} sqft")
                    st.write(f"**Bedrooms:** {int(beds)}")
                    st.write(f"**Bathrooms:** {int(bath)}")
                    st.write(f"**Floors:** {int(floor)}")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                with st.expander("🔍 Debug Information"):
                    st.write("**Input DataFrame:**")
                    st.dataframe(input_df)
                    st.write("**Error Details:**")
                    st.write(str(e))
    
    # Add footer
    st.write("---")
    st.caption("💡 Make sure your model and dataset files are in the same directory")

if __name__ == "__main__":
    main()