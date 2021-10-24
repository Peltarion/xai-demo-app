# XAI demo app

This is the repository for the explainability demo app.

It builds a webapp that asks a multi-class emotion classifier for predictions on short texts, and explains the predicitons using [SHAP](https://github.com/slundberg/shap), as shown in the video below:



https://user-images.githubusercontent.com/43994769/138614203-40aebbd3-a158-43b4-960e-db8820797e45.mp4



## To run the app: 

**1. Clone the repository:**
```
git clone git@github.com:Peltarion/xai-demo-app.git
cd xai-demo-app
```

**2. Store model and explainer in a pickle file:**
```
conda create --name env python=3.7
conda activate env 
pip install -r requirements.txt
python pickles.py
echo "pythonapi/ streamlitapi" | xargs -n 1 cp -v score_objects.pkl
```

**3. Spin up containers:**
```
docker-compose build
docker-compose up
```

**4. View the app in your browser at port 8501**

*Notes:*<br> 
You might need to change ports if the chosen ones are already allocated.<br> 
To use this repo as a template for a similar app, dump the components you need into a ```.pkl``` file by using a script similar to ```pickles.py```, and modify ```pythonapi/app.py``` and ```streamlitapi/webbapp.py``` accordingly. <br> 
