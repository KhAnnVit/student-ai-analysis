import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/AI_Student_Life_Pakistan_2026.csv')

# ПОДГОТОВКА И ОЧИСТКА ДАННЫХ

# Сколько пропусков в каждой колонке
print(df.isnull().sum())

'''
Вывод:

Student_ID            0
Age                   0
Gender                0
Education_Level       0
City                  0
AI_Tool_Used          0
Daily_Usage_Hours     0
Purpose               0
Impact_on_Grades      0
Satisfaction_Level    0

Итог: нет пропусков
'''

# Проверка на дубликаты
print(f"Дубликаты: {df.duplicated().sum()}")

'''
Вывод:

Дубликаты: 0

Итог: нет дубликатов
'''

# Проверка типов данных
print(df.dtypes)


'''
Вывод:

Student_ID              int64
Age                     int64
Gender                    str
Education_Level           str
City                      str
AI_Tool_Used              str
Daily_Usage_Hours     float64
Purpose                   str
Impact_on_Grades          str
Satisfaction_Level        str

Итог: с типами данных всё в порядке
'''

# Проверка уникальных значений(Ищем опечатки, лишние пробелы, разный регистр)
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}:")
    vc = df[col].value_counts()
    vc.index.name = None  # Убираем имя индекса
    print(vc)


'''
Вывод:

Gender:
Female    52
Male      48
Name: count, dtype: int64

Education_Level:
School        46
College       29
University    25
Name: count, dtype: int64

City:
Multan        29
Lahore        22
Islamabad     18
Faisalabad    17
Karachi       14
Name: count, dtype: int64

AI_Tool_Used:
Grammarly    24
Gemini       24
ChatGPT      20
Copilot      17
Notion AI    15
Name: count, dtype: int64

Purpose:
Homework    24
Learning    21
Research    21
Coding      20
Writing     14
Name: count, dtype: int64

Impact_on_Grades:
Improved          39
No Change         34
Slight Decline    27
Name: count, dtype: int64

Satisfaction_Level:
Low       38
Medium    32
High      30
Name: count, dtype: int64

Итог: все значения уникальны
'''

sns.histplot(df['Daily_Usage_Hours'], bins=20, kde=True)
plt.title('Распределение времени использования')
plt.xlabel('Часов в день')
plt.show()


'''
Итог: нет значительных выбросов. Есть несколько пиков (мультимодальное распределение)
'''