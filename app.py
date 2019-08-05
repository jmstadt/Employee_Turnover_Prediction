from fastai.tabular import *
from flask import Flask, request
import requests
import os.path


path = ''

export_file_url = 'https://www.dropbox.com/s/w55aizavo8ntqps/HR_data_Employee_Turnover_export.pkl?dl=1'
export_file_name = 'HR_data_Employee_Turnover_export.pkl'


def down_load_file(filename, url):
    """
    Download an URL to a file
    """
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)
            
def download_if_not_exists(filename, url):
    """
    Download a URL to a file if the file
    does not exist already.
    Returns
    -------
    True if the file was downloaded,
    False if it already existed
    """
    if not os.path.exists(filename):
        down_load_file(filename, url)
        return True
    return False

download_if_not_exists(export_file_name, export_file_url)

learn = load_learner(path, export_file_name)



app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':  #this block is only entered when the form is submitted
        
        satisfaction_level = request.form.get('satisfaction_level')
        
        last_evaluation = request.form.get('last_evaluation')
        
        number_project = request.form.get('number_project')
        
        average_montly_hours = request.form.get('average_montly_hours')
        
        time_spend_company = request.form.get('time_spend_company')
        
        Work_accident = request.form.get('Work_accident')
        
        promotion_last_5years = request.form.get('promotion_last_5years')
        
        sales = request.form.get('sales')
        
        salary = request.form.get('salary')
        
      
        
        inf_df = pd.DataFrame(columns=['satisfaction_level','last_evaluation', 'number_project', 
                                       'average_montly_hours', 'time_spend_company', 
                                       'Work_accident',
                                       'promotion_last_5years', 'sales', 'salary'])
        inf_df.loc[0] = [satisfaction_level, last_evaluation, number_project, average_montly_hours,
                         time_spend_company, Work_accident, 
                         promotion_last_5years, sales, salary]
        
        
        inf_df['satisfaction_level'] =  inf_df['satisfaction_level'].astype(float)
        inf_df['last_evaluation'] =  inf_df['last_evaluation'].astype(float)
        inf_df['number_project'] =  inf_df['number_project'].astype(int)
        inf_df['average_montly_hours'] =  inf_df['average_montly_hours'].astype(int)
        inf_df['time_spend_company'] =  inf_df['time_spend_company'].astype(int)
        inf_df['Work_accident'] =  inf_df['Work_accident'].astype(int)
        inf_df['promotion_last_5years'] =  inf_df['promotion_last_5years'].astype(int)
        
        
        inf_row = inf_df.iloc[0]
        
        pred = learn.predict(inf_row)
        
    
        
        return '''<h3>The input Satisfaction Level is: {}</h3>
                    <h3>The input Last Evaluation is: {}</h3>
                    <h3>The input Number of Projects is: {}</h3>
                    <h3>The input Average Monthly Hours are: {}</h3>
                    <h3>The input Number of Years Employed is: {}</h3>
                    <h3>The input on whether the employee had a work accident is (0 for No, 1 for Yes): {}</h3>
                    <h3>The input on whether the employee has been promoted in the last 5 years (0 for No, 1 for Yes) is: {}</h3>
                    <h3>The input Department is: {}</h3>
                    <h3>The input Salary Level is: {}</h3>
                    <h1>The employee risk of leaving (Category 1 for Yes, Category 0 for No): {}</h1>'''.format(satisfaction_level, 
                                                                                        last_evaluation, 
                                                                                        number_project,
                                                                                       average_montly_hours,
                                                                             time_spend_company,
                                                                             Work_accident,
                                                                                           promotion_last_5years,
                                                                                           sales, salary,
                                                                          pred
                                                                             )


    return '''<form method="POST">
                  <h1>Predicting whether an employee is at risk of leaving</h1>
                  
                  Satisfaction Level, enter a two digit decimal number between 0 and 1: <input type="number" name="satisfaction_level" step=0.01 min=0 max =1 required="required"><br>
                  
                  Last Evaluation, enter a two digit decimal number between 0 and 1: <input type="number" name="last_evaluation" step=0.01 min=0 max =1 required="required"><br>
                  
                  How many projects has the employee worked? enter a number between 0 and 9: <input type="number" name="number_project" step=1 min=0 max =9 required="required"><br>
                  
                  What is the average number of hours the employee works per month? <input type="number" name ="average_montly_hours" step=1 min=1 max=310 required="required"<br><br>
                  
                  How many years has the employee been employed? <input type="number" name ="time_spend_company" step=1 min=0 max=50 required="required"<br><br>
                  
                  Has the employee had a work accident (0 for No, 1 for Yes)? <input type="number" name ="Work_accident" step=1 min=0 max=1 required="required"<br><br>
                  
                  Has the employee been promoted in the last 5 years? (0 for No, 1 for Yes) <input type="number" name ="promotion_last_5years" step=1 min=0 max=1 required="required"<br><br>
                  
                  Select the employee's Department: <select name="sales">
                  <option value="sales">sales</option>
                  <option value="accounting">accounting</option>
                  <option value="hr">hr</option>
                  <option value="technical">technical</option>
                  <option value="support">support</option>
                  <option value="management">management</option>
                  <option value="IT">IT</option>
                  <option value="product_mng">product_mng</option>
                  <option value="marketing">marketing</option>
                  <option value="RandD">RandD</option>
                  </select><br>
                  
                  Is the employee's salary low, medium, or high?: <select name="salary">
                  <option value="low">low</option>
                  <option value="medium">medium</option>
                  <option value="high">high</option>
                  </select><br>
                  
                  <input type="submit" value="Submit"><br>
              </form>'''



#if __name__ == '__main__':
#    app.run(port=5000, debug=False)
