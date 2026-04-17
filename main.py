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
for col in df.select_dtypes(include=['object', 'str']).columns:
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
# plt.show()


'''
Итог: нет значительных выбросов. Есть несколько пиков (мультимодальное распределение)
'''


total = len(df)
tool_counts = df['AI_Tool_Used'].value_counts()
tool_pct = (tool_counts / total * 100).round(1)

print("Популярность AI-инструментов:")
for tool, count, pct in zip(tool_counts.index, tool_counts.values, tool_pct.values):
    print(f"  {tool:12} {count:3} человек ({pct}%)")



# Основные метрики
print("Время использования AI (часов в день):")
print(f"  Среднее:   {df['Daily_Usage_Hours'].mean():.2f}")
print(f"  Медиана:   {df['Daily_Usage_Hours'].median():.2f}")
print(f"  Минимум:   {df['Daily_Usage_Hours'].min():.2f}")
print(f"  Максимум:  {df['Daily_Usage_Hours'].max():.2f}")
print(f"  Стандартное отклонение: {df['Daily_Usage_Hours'].std():.2f}")

# Процентили
print(f"  25% студентов тратят ≤ {df['Daily_Usage_Hours'].quantile(0.25):.2f} ч")
print(f"  50% студентов тратят ≤ {df['Daily_Usage_Hours'].quantile(0.50):.2f} ч (медиана)")
print(f"  75% студентов тратят ≤ {df['Daily_Usage_Hours'].quantile(0.75):.2f} ч")
print(f"  90% студентов тратят ≤ {df['Daily_Usage_Hours'].quantile(0.90):.2f} ч")

# Количество и процент по каждому уровню
total = len(df)
sat_counts = df['Satisfaction_Level'].value_counts()
sat_pct = (sat_counts / total * 100).round(1)

print("Удовлетворённость использованием ИИ:")
for level, count, pct in zip(sat_counts.index, sat_counts.values, sat_pct.values):
    print(f"  {level:10} {count:3} студента ({pct:.1f}%)")




# Находим Топ-25% самых активных
threshold = df['Daily_Usage_Hours'].quantile(0.75)  # 75-й процентиль
active_users = df[df['Daily_Usage_Hours'] >= threshold]

print(f"Топ-25% активных (≥{threshold:.1f} ч/день): {len(active_users)} человек")


# Исследуем самых активных пользователей
print("\nПОРТРЕТ АКТИВНОГО ПОЛЬЗОВАТЕЛЯ:")

print(f"\nВозраст:")
print(f"  Средний: {active_users['Age'].mean():.1f} лет")
print(f"  Диапазон: {active_users['Age'].min()}–{active_users['Age'].max()} лет")

print(f"\nПол:")
print(active_users['Gender'].value_counts().apply(lambda x: f"{x} ({x/len(active_users)*100:.1f}%)"))

print(f"\nОбразование:")
print(active_users['Education_Level'].value_counts().apply(lambda x: f"{x} ({x/len(active_users)*100:.1f}%)"))

print(f"\nЛюбимые инструменты(топ-3):")
print(active_users['AI_Tool_Used'].value_counts().head(3).apply(lambda x: f"{x} человек  ({x/len(active_users)*100:.1f}%)"))

print(f"\nОсновные цели(топ-3):")
print(active_users['Purpose'].value_counts().head(3).apply(lambda x: f"{x} человек  ({x/len(active_users)*100:.1f}%)"))

print(f"\nВлияние на оценки:")
impact_pct = active_users['Impact_on_Grades'].value_counts(normalize=True) * 100
for impact, pct in impact_pct.items():
    print(f"  {impact}: {pct:.1f}%")

print(f"\nУдовлетворённость:")
sat_pct = active_users['Satisfaction_Level'].value_counts(normalize=True) * 100
for sat, pct in sat_pct.items():
    print(f"  {sat}: {pct:.1f}%")


# Фильтр: только недовольные
unhappy_users = df[df['Satisfaction_Level'] == 'Low']

print(f"Недовольных пользователей: {len(unhappy_users)} человек ({len(unhappy_users)/len(df)*100:.1f}%)")

# Обрабатываем недовольных пользователей
print("\nПОРТРЕТ НЕДОВОЛЬНОГО ПОЛЬЗОВАТЕЛЯ:")

print(f"\nВозраст:")
print(f"  Средний: {unhappy_users['Age'].mean():.1f} лет")
print(f"  Диапазон: {unhappy_users['Age'].min()}–{unhappy_users['Age'].max()} лет")

print(f"\nПол:")
gender_dist = unhappy_users['Gender'].value_counts()
for gender, count in gender_dist.items():
    pct = count / len(unhappy_users) * 100
    print(f"  {gender}: {count} ({pct:.1f}%)")

print(f"\nОбразование:")
edu_dist = unhappy_users['Education_Level'].value_counts()
for edu, count in edu_dist.items():
    pct = count / len(unhappy_users) * 100
    print(f"  {edu}: {count} ({pct:.1f}%)")


print(f"\nВремя использования:")
print(f"  Среднее: {unhappy_users['Daily_Usage_Hours'].mean():.2f} ч/день")
print(f"  Медиана: {unhappy_users['Daily_Usage_Hours'].median():.2f} ч/день")

print(f"\nИнструменты:")
tools = unhappy_users['AI_Tool_Used'].value_counts().head(3)
for tool, count in tools.items():
    print(f"  {tool}: {count} студента  ({len(unhappy_users)/len(df)*100:.1f}%)")

print(f"\nЦели использования:")
purposes = unhappy_users['Purpose'].value_counts().head(3)
for purpose, count in purposes.items():
    print(f"  {purpose}: {count} студента  ({len(unhappy_users)/len(df)*100:.1f}%)")

print(f"\nВлияние на оценки:")
impact = unhappy_users['Impact_on_Grades'].value_counts(normalize=True) * 100
for level, pct in impact.items():
    print(f"  {level}: {pct:.1f}%")




# === КРУГОВАЯ ДИАГРАММА: Популярность ИИ-инструментов ===

# 2. Подготавливаем данные
tool_counts = df['AI_Tool_Used'].value_counts()
total = len(df)

# 3. Задаём цветовую палитру
colors = ['#f94144', '#f8961e', '#f9c74f', '#90be6d', '#4d908e']

# 4. Создаём холст
plt.figure(figsize=(8, 8))

# 5. Рисуем круговую диаграмму
wedges, texts, autotexts = plt.pie(
    tool_counts.values,
    labels=tool_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    explode=[0.025] * len(tool_counts),
    textprops={'fontsize': 11}
)

# Улучшаем читаемость процентов: делаем их белыми и жирными
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Добавляем заголовок
plt.title(
    'Доля использования ИИ-инструментов',
    fontsize=14,
    fontweight='bold',
    pad=20
)

# 8. Автоматически подгоняем отступы, чтобы ничего не обрезалось
plt.tight_layout()

# 9. Сохраняем график в файл
plt.savefig(
    'images/tools_pie.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

# 10. Показываем график в отдельном окне
plt.show()

# 11. Печатаем подтверждение в консоль
print("✅ График сохранён в images/tools_pie.png")