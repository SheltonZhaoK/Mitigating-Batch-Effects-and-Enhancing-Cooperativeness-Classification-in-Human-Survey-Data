import os, argparse, json, matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

colors = ["#046586", "#28A9A1", "#C9A77C", "#F4A016",'#F6BBC6','#E71F19', "#9F2B68"]
customPalette = sns.color_palette(colors)
matplotlib.rcParams['svg.fonttype'] = 'none'
plt.rcParams["font.family"] = "Times"

def augmentation_scatter(train_data, train_labels, label, args, configs):
    plt.figure(figsize=(6.4, 4.8))
    if not args.d == "raw":
        plot_data = pd.concat([train_data, train_labels], axis=1)
        sns.scatterplot(data=plot_data, x=plot_data.columns[0], y=plot_data.columns[1], hue=label, palette = customPalette)
        plt.title(f'{args.d} with {args.a} and {args.gan}')
        print("output figure")
        # plt.show()
        plt.savefig(os.path.join(configs["outputDir"], args.e, f"{args.a}_{args.d}_{args.t}_{args.gan}_scaterPlot.svg"))

def preliminary_scatter(train_data, train_labels, label, target, args, configs):
    if not args.d == "raw":
        plot_data = pd.concat([train_data, target], axis=1)
        plot_data = pd.concat([plot_data, train_labels], axis=1)
        plt.figure(figsize=(6.4*3, 4.8))  # Adjust the size as needed
        plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st subplot
        sns.scatterplot(data=plot_data, x=plot_data.columns[0], y=plot_data.columns[1], hue="RWAVENEW", palette = customPalette)
        plt.title(f'{args.d} with RWAVENEW')
        plt.subplot(1, 3, 2)  # 1 row, 2 columns, 2nd subplot
        sns.scatterplot(data=plot_data, x=plot_data.columns[0], y=plot_data.columns[1], hue="RWAVEOLD", palette = customPalette )
        plt.title(f'{args.d} with RWAVEOLD')
        plt.subplot(1, 3, 3)  # 1 row, 2 columns, 2nd subplot
        sns.scatterplot(data=plot_data, x=plot_data.columns[0], y=plot_data.columns[1], hue=label, palette = customPalette )
        plt.title(f'{args.d} with label')
        plt.savefig(os.path.join(configs["outputDir"], args.e, f"D({args.d})_scaterPlot.svg"))

def result1():
    data = pd.read_csv("../output/preliminary/preliminary_results.csv")
    data.fillna('normal', inplace=True)
    plt.figure(figsize=(6.4*2, 4.8))
    i = 1
    for metric in ["Accuracy", "Weighted F1"]:
        plt.subplot(1, 2, i)  # 1 row, 2 columns, 1st subplot
        sns.barplot(data=data, x="Data Type", y=metric, hue="Classifiers", palette = customPalette)
        i += 1
    plt.savefig("../output/result1A.svg")

    plt.figure(figsize=(6.4*3, 4.8))
    for i, tree in enumerate(["decisionTree", "xgb", "randomForest"]):
        with open(f"../output/preliminary/D(raw)_T(None)_A(None)_{tree}_S(1).json", 'r') as file:
            data = json.load(file)
            df = pd.DataFrame(data.items(), columns=['Feature', 'Feature Importance']).sort_values(by='Feature Importance', ascending=False)
            df = df.head(10)
            plt.subplot(1, 3, i+1)
            sns.barplot(x='Feature Importance', y='Feature', data=df)
            plt.title(f'{tree}_Feature Importance Plot')
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.tight_layout()
    plt.savefig("../output/result1B.svg")


def result2():
    # data = pd.read_csv("../output/baseline/baseline_results.csv")
    # data.fillna('normal', inplace=True)
    # plt.figure(figsize=(6.4, 4.8*2))
    # i = 1

    # for tempt in [data[data["Data Transformation"] == "normal"], data[data["Data Transformation"] == "oneW"]]:
    #     for metric in ["Accuracy"]:
    #         plt.subplot(2,1,i)
    #         sns.barplot(data=tempt, x="Data Type", y=metric, hue="Classifiers", palette = customPalette)
    #         i += 1
    # plt.savefig("../output/result2A.svg")

    # report = pd.DataFrame(columns = ["Data Type", "transformation", "Classifiers", "Class 0 Accuracy", "Class 1 Accuracy"])
    # for transformation in [None, "oneW"]:
    #     for dataType in ["raw", "pca", "umap", "pca_umap"]:
    #         dataPath = f"../output/baseline/ANone_D{dataType}_T{transformation}_GFalse_testingData_predications.csv"
    #         data = pd.read_csv(dataPath)
    #         labels0 = data[data["label"] == 0] #fake response
    #         labels1 = data[data["label"] == 1] #real response
    #         for classifier in ["naive","MLP","logisticRegression","decisionTree","randomForest","xgb"]:
    #             class0_accuracy = (labels0[classifier] == 0).mean()
    #             class1_accuracy = (labels1[classifier] == 1).mean()
    #             report.loc[len(report)] = [dataType, transformation, classifier, class0_accuracy, class1_accuracy]

    # report.fillna('normal', inplace=True)
    # report.to_csv("../output/baseline_classAccuracy_report.csv")
    # plt.figure(figsize=(6.4*2, 4.8*2))
    # i = 1
    # for tempt in [report[report["transformation"] == "normal"], report[report["transformation"] == "oneW"]]:
    #     for metric in ["Class 0 Accuracy", "Class 1 Accuracy"]:
    #         plt.subplot(2, 2, i)
    #         sns.barplot(data=tempt, x="Data Type", y=metric, hue="Classifiers", palette = customPalette)
    #         plt.ylim(0, 1.0)
    #         i += 1
    # plt.savefig("../output/result2B.svg")

    # plt.figure(figsize=(6.4*3, 4.8*2))
    # for i, tree in enumerate(["decisionTree", "xgb", "randomForest"]):
    #     with open(f"../output/baseline/D(raw)_T(None)_A(None)_{tree}_S(1).json", 'r') as file:
    #         data = json.load(file)
    #         df = pd.DataFrame(data.items(), columns=['Feature', 'Feature Importance']).sort_values(by='Feature Importance', ascending=False)
    #         df = df.head(10)
    #         plt.subplot(2, 3, i+1)
    #         sns.barplot(x='Feature Importance', y='Feature', data=df)
    #         plt.title(f'{tree}_Feature Importance Plot')
    #         plt.xlabel('Feature Importance')
    #         plt.ylabel('Features')
    #         plt.tight_layout()

    # for i, tree in enumerate(["decisionTree", "xgb", "randomForest"]):
    #     with open(f"../output/baseline/D(raw)_T(oneW)_A(None)_{tree}_S(1).json", 'r') as file:
    #         data = json.load(file)
    #         df = pd.DataFrame(data.items(), columns=['Feature', 'Feature Importance']).sort_values(by='Feature Importance', ascending=False)
    #         df = df.head(10)
    #         plt.subplot(2, 3, i+4)
    #         sns.barplot(x='Feature Importance', y='Feature', data=df)
    #         plt.title(f'{tree}_Feature Importance Plot')
    #         plt.xlabel('Feature Importance')
    #         plt.ylabel('Features')
    #         plt.tight_layout()
    # plt.savefig("../output/result2C.svg")

    training_data = "../output/baseline/ANone_Dpca_TNone_GFalse_trainingData.csv"
    testing_data = "../output/baseline/ANone_Dpca_TNone_GFalse_testingData_predications.csv"
    training_data = pd.read_csv(training_data)
    testing_data = pd.read_csv(testing_data)
    plt.figure(figsize=(6.4*2, 4.8*1))
    plt.subplot(1, 2, 1)
    plt.ylim(-4, 4.5)
    sns.scatterplot(data=training_data, x=training_data.columns[1], y=training_data.columns[2], hue="W3IWER4_A", palette = customPalette)
    plt.subplot(1, 2, 2)
    plt.ylim(-4, 4.5)
    sns.scatterplot(data=testing_data, x=testing_data.columns[1], y=testing_data.columns[2], hue="label", palette = customPalette)
    plt.savefig("../output/result2D.svg")

def result3():
    baseline_path = '../output/baseline/baseline_results.csv'
    augmentation_path = '../output/augmentation/augmentation_results.csv'

    baseline_df = pd.read_csv(baseline_path)
    augmentation_df = pd.read_csv(augmentation_path)
    baseline_df.fillna('normal', inplace=True)
    augmentation_df.fillna('normal', inplace=True)

    report = pd.DataFrame(columns = ["Augmentation Methods", "GAN", "Data Type", "Data Transformation", "Classifiers", "Improvement in Accuracy"])
    for augmentation in ["smote", "editNN", "tomkLink", "smoteNN", "smoteTomek"]:
        for gan in [True, False]:
            for dataType in ["raw", "pca", "umap", "pca_umap"]:
                for transformation in ["normal", "oneW"]:
                    for classifier in ["MLP", "logisticRegression", "decisionTree", "randomForest", "xgb"]:
                        temp_baseline_df = baseline_df[
                            (baseline_df["Data Type"] == dataType) & 
                            (baseline_df["Data Transformation"] == transformation) & 
                            (baseline_df["Classifiers"] == classifier)
                        ]

                        temp_augmentation_df = augmentation_df[
                            (augmentation_df["Data Augmentation Methods"] == augmentation) & 
                            (augmentation_df["GAN"] == gan) & 
                            (augmentation_df["Data Type"] == dataType) & 
                            (augmentation_df["Data Transformation"] == transformation) & 
                            (augmentation_df["Classifiers"] == classifier)
                        ]
                        improvement = temp_augmentation_df.iloc[0]["Accuracy"] - temp_baseline_df.iloc[0]["Accuracy"]
                        report.loc[len(report)] = [augmentation, gan, dataType, transformation, classifier, round(improvement, 4)]
    report.to_csv("../output/augmentation_improvement.csv")

    plt.figure(figsize=(6.4*5, 4.8*4))
    i = 1
    for gan in [False, True]:
        for transformation in ["normal", "oneW"]:
            for augmentation in ["smote", "editNN", "tomkLink", "smoteNN", "smoteTomek"]:
                tempt_data = report[
                    (report["GAN"] == gan) & 
                    (report["Data Transformation"] == transformation) & 
                    (report["Augmentation Methods"] == augmentation)]
                plt.subplot(4, 5, i)
                sns.barplot(data=tempt_data, x="Data Type", y="Improvement in Accuracy", hue="Classifiers", palette = customPalette)
                plt.title(f'{augmentation}_{gan}')
                plt.ylim(-0.4, 0.3)
                plt.xticks([0,1,2,3],["Raw", "PCA", "UMAP", "PCA_UMAP"])
                # plt.legend().set_visible(False)
                i += 1
    plt.savefig("../output/result3.svg")

    # augmentation_df = augmentation_df[augmentation_df["Classifiers"] != "naive"]
    # plt.figure(figsize=(6.4*5, 4.8*4))
    # i = 1
    # for gan in [False, True]:
    #     for transformation in ["normal", "oneW"]:
    #         for augmentation in ["smote", "editNN", "tomkLink", "smoteNN", "smoteTomek"]:
    #             tempt_data = augmentation_df[
    #                 (augmentation_df["GAN"] == True) & 
    #                 (augmentation_df["Data Transformation"] == transformation) & 
    #                 (augmentation_df["Data Augmentation Methods"] == augmentation)]
    #             plt.subplot(4, 5, i)
    #             sns.barplot(data=tempt_data, x="Data Type", y="Accuracy", hue="Classifiers", palette = customPalette)
    #             plt.title(f'{augmentation}_{gan}')
    #             plt.ylim(0, 0.8)
    #             i += 1
    # plt.savefig("../output/result3B.svg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dataMining assignment5')
    parser.add_argument('-e', type=str, default=None, help='specify which result to generate')
    args = parser.parse_args()

    if args.e == "result1":
        result1()
    elif args.e == "result2":
        result2()
    elif args.e == "result3":
        result3()
