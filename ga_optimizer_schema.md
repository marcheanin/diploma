```mermaid
graph TD
    A[ga_optimizer.py];
    B(ИНИЦИАЛИЗАЦИЯ И КОНФИГУРАЦИЯ);
    C{run_genetic_algorithm()};
    MainLoopEntry(Начало главного цикла);
    MainLoopExit(Конец главного цикла);
    E(АНАЛИЗ РЕЗУЛЬТАТОВ И ОТЧЕТНОСТЬ);

    A --> B;
    B --> C;
    C --> MainLoopEntry;
    MainLoopExit --> E;

    subgraph S_MainLoop [Главный Цикл ГА]
        direction LR
        MainLoopEntry --> C1[Создание начальной популяции];
        C1 -.-> D_init_pop[initialize_population()];
        C1 --> C2{ЦИКЛ ПО ГЕНЕРАЦИЯМ};

        subgraph C2 [Генерация]
            direction TB
            C2_Eval[\n1. ОЦЕНКА ОСОБЕЙ\n(evaluate_chromosome)\n];
            C2_Eval -.-> D_eval[evaluate_chromosome()];
            D_eval --> D_decode[main.decode_and_log_chromosome()];
            D_eval --> D_proc[pipeline_processor.process_data()];
            D_eval --> D_train[main.train_model()];
            D_decode --> D_eval_res[(Фитнес, Тип модели)];
            D_proc --> D_eval_res;
            D_train --> D_eval_res;

            C2_Eval --> C2_Update[2. ОБНОВЛЕНИЕ ЛУЧШИХ РЕЗУЛЬТАТОВ];
            C2_Update --> C2_NextGen_Group[3. ФОРМИРОВАНИЕ СЛЕД. ПОКОЛЕНИЯ];
            
            subgraph C2_NextGen_Group [Формирование Поколения]
                direction TB
                NG1[Элитизм];
                NG1 --> NG2{Цикл создания потомков};
                subgraph NG2 [Создание Потомков]
                    direction TB
                    P_Sel[Селекция родителей] -.-> D_tour[select_parent_tournament()];
                    P_Sel --> P_Cross[Скрещивание] -.-> D_cross[crossover()];
                    P_Cross --> P_Mut[Мутация] -.-> D_mut[mutate()];
                end
            end
            C2_NextGen_Group --> C2_PopUpdate[4. Обновление популяции];
            C2_PopUpdate --> MainLoopExit;
        end
    end

    E --> E_Best[Вывод лучшей хромосомы и фитнеса];
    E --> E_Decode_Best[decode_and_log_chromosome() для лучшей];
    E --> E_Plot[Построение и сохранение графика (matplotlib)];

    classDef mainFunctions fill:#f9f,stroke:#333,stroke-width:2px;
    classDef subProcess fill:#bbf,stroke:#333,stroke-width:1px;
    classDef libCall fill:#lightgrey,stroke:#333,stroke-width:1px,color:black;
    classDef invisible stroke-width:0,color:#00000000,fill:#00000000;

    class C,C2_Eval,P_Sel,P_Cross,P_Mut mainFunctions;
    class B,E,C1,C2_Update,C2_NextGen_Group,C2_PopUpdate,NG1,NG2,E_Best,E_Decode_Best,E_Plot,MainLoopEntry subProcess;
    class D_init_pop,D_eval,D_decode,D_proc,D_train,D_tour,D_cross,D_mut libCall;
    class MainLoopExit invisible;
``` 