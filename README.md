# Objetivo
A partir da base de dados British Birdsong Dataset, temos como ideia criar um classificador de espécies a partir dos dados sonoros. A base possui um arquivo de meta-dados, sendo possível verificar o nome do arquivo, género, espécie e demais informações geográficas.
Mais informações no site oficial: https://archive.org/details/xccoverbl_2014 uma vez com um modelo bem acurado temos o objetivo de desemvolver uma API que retorn classificação de especie de passaro pelo canto e possa servir estes resultado em um front end

# Metodologia
Para criar um classificador iremos usar a abordagem de recuperação de informações musicais, comummente referenciada pelo termo em inglês Music Information Retrieval (MIR), é um emergente campo de pesquisa que trata da recuperação e organização de grandes coleções ou informações musicais, de acordo com sua relevância para consultas específicas. Esta prática tem se tornado extremamente relevante, dada a vasta quantidade de informações e serviços relacionados a música que existem atualmente mas podemos utilizar a mesma ideia para capturar informações do canto de cada espécie de pássaros as informações que são capturadas nada mais são do que chromogramas, tempogramas e são baseados em frequências sonora mais informações na documentação do librosa biblioteca que usamos para esta finalidade: https://librosa.org/doc/main/feature.html

Uma vez que temos os nossos dados estruturados de forma tabular podemos aplicar feature engineer, feature select, técnicas de machine learning e validações robusta através de cross validate, k-fold e métricas de classificação.

# Status atual do Projeto
 Apos os primeiros esforces nesta pesquisa não obtivemos resultados satisfatórios no entanto podemos observar abordagens que ainda precisaríamos testar em primeiro lugar uma abordagem para pegar somente o canto dos pássaros e retirar com mais exatidão ruídos externos do ambiente mas principalmente mais exemplos de dados para cada especie. Explorar modelos que são mais robustos para problemas não linearmente separáveis como arquiteturas de redes neurais e modelos eurísticos como algoritmos genéticos e buscar conhecer novas features no campo de MIR e testar transformações matemáticas nas features mais importantes descobertas. Os melhores resultado para cada modelo foram:
 
 # Academicos 
 Douglas silva
 Leandro de Jesus 
