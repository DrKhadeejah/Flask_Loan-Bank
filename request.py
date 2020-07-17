import requests

url ='http://localhost:5000/results'
r = requests.post(url,json = {'ApplicantIncome': 4000, 'CoapplicantIncome': 2000, 'LoanAmount': 60000, 'Self_Employed': 0, 'Loan_Amount_Term': 360})

print(r.json())