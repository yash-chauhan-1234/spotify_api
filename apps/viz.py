import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import script

def app():

    st.write("You need to run either Playlist or Recommend once before running this")
    df_fav=script.df_main
    # st.dataframe(df_fav)
    try:
        with st.expander("Sample Visualizations:"):
                c1,c2=st.columns(2)
                fig1=plt.figure()
                plt.title("Energy plot")
                sns.distplot(df_fav["energy"])
                c1.pyplot(fig1)

                fig2=plt.figure()
                plt.title("Loudness Graph")
                sns.boxplot(data=df_fav, x="loudness", orient="h")
                c2.pyplot(fig2)

                c3, c4=st.columns(2)
                audio_corr=df_fav.drop(["name", "id", "mode"], axis=1).corr()
                fig3=plt.figure()
                plt.title("Correlation plot")
                sns.heatmap(audio_corr)
                c3.pyplot(fig3)

                fig4=plt.figure()
                plt.title("Energy vs Loudness")
                sns.scatterplot(data=df_fav, x="energy", y="loudness")
                c4.pyplot(fig4)

        with st.form("Custom Visualizations"):
            x_axis=st.text_input(label="Enter the X-Axis field")
            y_axis=st.text_input(label="Enter the Y-Axis field")
            submit=st.form_submit_button(label="Submit")

            if submit:
                st.markdown("---")

                fig5=plt.figure()
                if x_axis==y_axis:
                    plt.title(x_axis+" histogram")
                    sns.histplot(data=df_fav, x=x_axis)
                else:
                    plt.title(y_axis+" vs "+x_axis)
                    sns.scatterplot(data=df_fav, x=x_axis, y=y_axis)
                st.pyplot(fig5)
            # st.markdown("[Click here]() for a more comprehensive visualization!")
    except:
        st.write("You need to run Recommend or Playlist before running this")