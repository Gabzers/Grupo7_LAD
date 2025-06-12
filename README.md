# Grupo7_Trabalho
 Trabalho de LAD


Challenges: (10 %) 
• Employ  recommendation  algorithms  to  your  dataset  when possible.  If  it is not 
possible you should justify why. 




 Analisando o seu projeto no #codebase, vejo que vocês estão trabalhando com classificação de tipos de ataques em redes IoT usando diversos algoritmos de machine learning como Random Forest, SVM, KNN, etc.

Resposta: NÃO é diretamente aplicável
Os algoritmos de recomendação não são adequados para o seu projeto atual pelas seguintes razões:

Por que não se aplica:
Natureza do problema: Vocês têm um problema de classificação supervisionada (predizer tipos de ataques), não de recomendação

Estrutura dos dados: Seus dados são características técnicas de rede (features numéricas de tráfego IoT) com labels de tipos de ataques, não dados de preferências usuário-item

Objetivo: O goal é detectar/classificar ataques, não recomendar itens para usuários

Justificativa técnica:
Os algoritmos de recomendação (Collaborative Filtering, Content-Based, Matrix Factorization) requerem:

Dados de interação usuário-item
Ratings ou preferências implícitas/explícitas
Objetivo de sugerir novos itens
Seu dataset RT_IOT2022.csv contém features de rede IoT para classificação binária/multiclasse de ataques.

Alternativa válida:
Se quiserem incluir algoritmos de recomendação, poderiam criar um cenário artificial:

"Sistema de recomendação de contramedidas de segurança baseado em tipos de ataques detectados"
Recomendar ações de segurança específicas para cada tipo de ataque identificado
Mas isso seria uma extensão forçada do projeto original, não uma aplicação natural dos dados existentes.

Conclusão: É perfeitamente justificável dizer que algoritmos de recomendação não se aplicam ao seu problema de detecção de ataques IoT.
