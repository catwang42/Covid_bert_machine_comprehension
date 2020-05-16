

const fs = require('fs');
console.log("test")
//const DATA_DIR = '/Users/pablo/Desktop/jupyter_notebooks/stanford_NLU/cs224u/data/bioasq/BioASQ-test8b'; 
const DATA_DIR = 'D:\\stanford_courses\\nlu-all\\final_project\\cs224u\\dev_notebook\\finetune\\bioasq_data\\train';
const filesPaths = [
    DATA_DIR + '\\training8b_squad_format_300.json',
    DATA_DIR + '\\training8b_squad_format_600.json',
    DATA_DIR + '\\training8b_squad_format_900.json',
    DATA_DIR + '\\training8b_squad_format_1200.json',
    DATA_DIR + '\\training8b_squad_format_1500.json',
    DATA_DIR + '\\training8b_squad_format_1800.json',
    DATA_DIR + '\\training8b_squad_format_2100.json',
    DATA_DIR + '\\training8b_squad_format_2400.json',
    DATA_DIR + '\\training8b_squad_format_2700.json',
    DATA_DIR + '\\training8b_squad_format_3000.json',
    DATA_DIR + '\\training8b_squad_format_3243.json',
];
const outputFilename = DATA_DIR + '\\trainset_combined_squad_format.json'

// let datasets = [
//   {
//     version: 'BioASQ8b',
//     data: [
//       {
//         title: 'summary',
//         paragraphs: [
//           {
//             qas: [
//               {
//                 answers: [
//                   {
//                     text:
//                       'Compound 12c showed potent and prolonged LH suppression after a single dose was orally administered in castrated monkeys compared to a known antagonist, Elagolix. ',
//                     answer_start: 604,
//                   },
//                 ],
//                 question:
//                   'Describe the mechanism of action of a drug Elagolix.',
//                 id: '5e2900368b3851296d000001_001',
//                 is_impossible: false,
//               },
//             ],
//             context:
//               'We investigated a series of uracil analogues by introducing various substituents on the phenyl ring of the N-3 aminoethyl side chain and evaluated their antagonistic activity against human gonadotropin-releasing hormone (GnRH) receptors. Analogues with substituents at the ortho or meta position demonstrated potent in vitro antagonistic activity. Specifically, the introduction of a 2-OMe group enhanced nuclear factor of activated T-cells (NFAT) inhibition up to 6-fold compared to the unsubstituted analogue. We identified compound 12c as a highly potent GnRH antagonist with moderate CYP inhibition. Compound 12c showed potent and prolonged LH suppression after a single dose was orally administered in castrated monkeys compared to a known antagonist, Elagolix. We believe that our SAR study offers useful insights to design GnRH antagonists as a potential treatment option for endometriosis.',
//           },
//           {
//             qas: [
//               {
//                 answers: [
//                   {
//                     text:
//                       'INTRODUCTION: Elagolix is a novel, orally active, non-peptide, competitive gonadotropin-releasing hormone (GnRH) receptor antagonist in development for the management of endometriosis with associated pain and heavy menstrual bleeding due to uterine fibroids. ',
//                     answer_start: 0,
//                   },
//                 ],
//                 question:
//                   'Describe the mechanism of action of a drug Elagolix.',
//                 id: '5e2900368b3851296d000001_002',
//                 is_impossible: false,
//               },
//             ],
//             context:
//               'INTRODUCTION: Elagolix is a novel, orally active, non-peptide, competitive gonadotropin-releasing hormone (GnRH) receptor antagonist in development for the management of endometriosis with associated pain and heavy menstrual bleeding due to uterine fibroids. The pharmacokinetics of elagolix have been well-characterized in phase I studies; however, elagolix population pharmacokinetics have not been previously reported. Therefore, a robust model was developed to describe elagolix population pharmacokinetics and to evaluate factors affecting elagolix pharmacokinetic parameters.METHODS: The data from nine clinical studies (a total of 1624 women) were included in the analysis: five phase I studies in healthy, premenopausal women and four phase III studies in premenopausal women with endometriosis.RESULTS: Elagolix population pharmacokinetics were best described by a two-compartment model with a lag time in absorption. Of the 15 covariates tested for effect on elagolix apparent clearance (CL/F) and/or volume of distribution only one covariate, organic anion transporting polypeptide (OATP) 1B1 genotype status, had a statistically significant, but not clinically meaningful, effect on elagolix CL/F.CONCLUSION: Elagolix pharmacokinetics were not affected by patient demographics and were similar between healthy women and women with endometriosis. Clinical Trial Registration Numbers NCT01403038, NCT01620528, NCT01760954, NCT01931670, NCT02143713.',
//           },
//           {
//             qas: [
//               {
//                 answers: [
//                   {
//                     text:
//                       'OBJECTIVE: To evaluate the efficacy and safety of elagolix, an oral, nonpeptide gonadotropin-releasing hormone antagonist, over 12 months in women with endometriosis-associated pain.',
//                     answer_start: 0,
//                   },
//                 ],
//                 question:
//                   'Describe the mechanism of action of a drug Elagolix.',
//                 id: '5e2900368b3851296d000001_003',
//                 is_impossible: false,
//               },
//             ],
//             context:
//               'OBJECTIVE: To evaluate the efficacy and safety of elagolix, an oral, nonpeptide gonadotropin-releasing hormone antagonist, over 12 months in women with endometriosis-associated pain.METHODS: Elaris Endometriosis (EM)-III and -IV were extension studies that evaluated an additional 6 months of treatment after two 6-month, double-blind, placebo-controlled phase 3 trials (12 continuous treatment months) with two elagolix doses (150 mg once daily and 200 mg twice daily). Coprimary efficacy endpoints were the proportion of responders (clinically meaningful pain reduction and stable or decreased rescue analgesic use) based on average monthly dysmenorrhea and nonmenstrual pelvic pain scores. Safety assessments included adverse events, clinical laboratory tests, and endometrial and bone mineral density assessments. The power of Elaris EM-III and -IV was based on the comparison to placebo in Elaris EM-I and -II with an expected 25% dropout rate.RESULTS: Between December 28, 2012, and October 31, 2014 (Elaris EM-III), and between May 27, 2014, and January 6, 2016 (Elaris EM-IV), 569 participants were enrolled. After 12 months of treatment, Elaris EM-III responder rates for dysmenorrhea were 52.1% at 150 mg once daily (Elaris EM-IV 550.8%) and 78.2% at 200 mg twice daily (Elaris EMIV 575.9%). Elaris EM-III nonmenstrual pelvic pain responder rates were 67.5% at 150 mg once daily (Elaris EM-IV 566.4%) and 69.1% at 200 mg twice daily (Elaris EM-IV 567.2%).”After 12 months of treatment, Elaris EM-III dyspareunia responder rates were 45.2% at 150 mg once daily (Elaris EM-IV=45.9%) and 60.0% at 200 mg twice daily (Elaris EM-IV=58.1%). Hot flush was the most common adverse event. Decreases from baseline in bone mineral density and increases from baseline in lipids were observed after 12 months of treatment. There were no adverse endometrial findings.CONCLUSION: Long-term elagolix treatment provided sustained reductions in dysmenorrhea, nonmenstrual pelvic pain, and dyspareunia. The safety was consistent with reduced estrogen levels and no new safety concerns were associated with long-term elagolix use.CLINICAL TRIAL REGISTRATION: ClinicalTrials.gov, NCT01760954 and NCT02143713.',
//           },
//           {
//             qas: [
//               {
//                 answers: [
//                   {
//                     text:
//                       'Phase III trials on elagolix, a new oral GnRH antagonist but non-inferiority RCT data are required to compare elagolix with first-line therapies for endometriosis.',
//                     answer_start: 1504,
//                   },
//                 ],
//                 question:
//                   'Describe the mechanism of action of a drug Elagolix.',
//                 id: '5e2900368b3851296d000001_004',
//                 is_impossible: false,
//               },
//             ],
//             context:
//               'Endometriosis is a chronic benign disease that affects women of reproductive age. Medical therapy is often the first line of management for women with endometriosis in order to ameliorate symptoms or to prevent post-surgical disease recurrence. Currently, there are several medical options for the management of patients with endometriosis. Non-steroidal anti-inflammatory drugs (NSAIDs) are widely used in the treatment of chronic inflammatory conditions, being efficacious in relieving primary dysmenorrhea. Combined oral contraceptives (COCs) and progestins, available for multiple routes of administration, are effective first-line hormonal options. In fact, several randomized controlled trials (RCTs) demonstrated that they succeed in improving pain symptoms in the majority of patients, are well tolerated and not expensive. Second-line therapy is represented by gonadotropin-releasing hormone (GnRH) agonists. Even if these drugs are efficacious in treating women not responding to COCs or progestins, they are not orally available and have a less favorable tolerability profile (needing an appropriate add-back therapy). The use of danazol is limited by the large availability of other better-tolerated hormonal drugs. Because few data are available on long-term efficacy and safety of aromatase inhibitors they should be administered only in women with symptoms refractory to other conventional therapies in a clinical research setting. Promising preliminary data have emerged from multicenter Phase III trials on elagolix, a new oral GnRH antagonist but non-inferiority RCT data are required to compare elagolix with first-line therapies for endometriosis.',
//           },
//           {
//             qas: [
//               {
//                 answers: [
//                   {
//                     text:
//                       'Elagolix (ORILISSA™), an orally bioavailable, second-generation, non-peptide gonadotropin-releasing hormone (GnRH) receptor antagonist, is being developed AbbVie and Neurocrine Biosciences for the treatment of reproductive hormone-dependent disorders in women.',
//                     answer_start: 0,
//                   },
//                 ],
//                 question:
//                   'Describe the mechanism of action of a drug Elagolix.',
//                 id: '5e2900368b3851296d000001_005',
//                 is_impossible: false,
//               },
//             ],
//             context:
//               'Elagolix (ORILISSA™), an orally bioavailable, second-generation, non-peptide gonadotropin-releasing hormone (GnRH) receptor antagonist, is being developed AbbVie and Neurocrine Biosciences for the treatment of reproductive hormone-dependent disorders in women. In July 2018, the US FDA approved elagolix tablets for the management of moderate to severe pain associated with endometriosis. This approval was based on positive results in two replicate phase III trials; additional phase III trials in the USA, Canada and Puerto Rico are currently evaluating elagolix as both monotherapy and in combination with low-dose hormone add-back therapy in the same indication. Elagolix with and without low-dose hormone add-back therapy is also undergoing phase III clinical development for heavy menstrual bleeding associated with uterine fibroids in the aforementioned locations. This article summarizes the milestones in the development of elagolix leading to its first approval for the management of moderate to severe pain associated with endometriosis.',
//           },
//           {
//             qas: [
//               {
//                 answers: [
//                   {
//                     text:
//                       'OBJECTIVE: To evaluate elagolix, an oral gonadotropin-releasing hormone receptor antagonist, alone or with add-back therapy, in premenopausal women with heavy menstrual bleeding (greater than 80 mL per month) associated with uterine leiomyomas.',
//                     answer_start: 0,
//                   },
//                 ],
//                 question:
//                   'Describe the mechanism of action of a drug Elagolix.',
//                 id: '5e2900368b3851296d000001_006',
//                 is_impossible: false,
//               },
//             ],
//             context:
//               'OBJECTIVE: To evaluate elagolix, an oral gonadotropin-releasing hormone receptor antagonist, alone or with add-back therapy, in premenopausal women with heavy menstrual bleeding (greater than 80 mL per month) associated with uterine leiomyomas.METHODS: This double-blind, randomized, placebo-controlled, parallel-group study evaluated efficacy and safety of elagolix in cohorts 1 (300 mg twice daily) and 2 (600 mg daily) with four arms per cohort: placebo, elagolix alone, elagolix with 0.5 mg estradiol/0.1 norethindrone acetate, and elagolix with 1.0 mg estradiol/0.5 mg norethindrone acetate. A sample size of 65 per group was planned to compare elagolix with add-back to placebo on the primary end point: the percentage of women who had less than 80 mL menstrual blood loss and 50% or greater reduction in menstrual blood loss from baseline to the last 28 days of treatment. Safety assessments included changes in bone mineral density.RESULTS: From April 8, 2013, to December 8, 2015, 571 women were enrolled, 567 were randomized and treated (cohort 1=259; cohort 2=308), and 80% and 75% completed treatment, respectively. Participants had a mean±SD age of 43±5 years (cohort 2, 42±5 years), and 70% were black (cohort 2, 74%). Primary end point responder rates in cohort 1 (cohort 2) were 92% (90%) for elagolix alone, 85% (73%) for elagolix with 0.5 mg estradiol/0.1 mg norethindrone acetate, 79% (82%) for elagolix with 1.0 mg estradiol/0.5 mg norethindrone acetate, and 27% (32%) for placebo (all P<.001 vs placebo). Elagolix groups had significant decreases compared with placebo in lumbar spine bone mineral density, which was attenuated by adding 1.0 mg estradiol/0.5 mg norethindrone acetate.CONCLUSION: Elagolix with and without add-back significantly reduced menstrual blood loss in women with uterine leiomyomas. Add-back therapy reduced hypoestrogenic effects on bone mineral density.CLINICAL TRIAL REGISTRATION: ClinicalTrials.gov, NCT01817530; EU Clinical Trial Register, 2013-000082-37.',
//           },
//         ],
//       },
//     ],
//   },
// ];
// datasets = datasets.concat(datasets)
let datasets = filesPaths.map((filePath) => require(filePath));
//console.log(datasets)

let newDataset = datasets.reduce((result, dataset) => {
  console.log('processing dataset length: ', dataset.data.length);
  if (!result['version']) {
    // result is empty
    result = dataset;
  } else {
    result.data = result.data.concat(dataset.data);
  }
  return result;
}, {});

console.log('final dataset length: ', newDataset.data.length);


var jsonContent = JSON.stringify(newDataset);
 
fs.writeFile(outputFilename, jsonContent, 'utf8', function (err) {
    if (err) {
        console.log("An error occured while writing JSON Object to File.");
        return console.log(err);
    }
    console.log("JSON file has been saved.");
});

