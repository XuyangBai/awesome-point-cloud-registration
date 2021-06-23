# awesome-point-cloud-registration

A curated list of resources on point cloud registration inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision). Work-in-progress. All contributions are welcome and appreciated.

This list focuses on the rigid registration between point clouds. 

## Table of Contents

- [Coarse Registration (Global Registration)](#coarse-registration)
    - [Feature Matching Based](#feature-matching-based)
        - [Keypoint Detection](#keypoint-detection)
        - [Feature Description](#feature-description)
        - [Outlier Rejection](#outlier-rejection)
        - [Graph Algorithms](#graph-algorithms)
    - [End-to-End](#end-to-end)
    - [Randomized](#randomized)
    - [Probablistic](#probabilistic)
- [Fine Registration (Local Registration)](#fine-registration)
    - [Learning-based](#learning-based)
    - [Traditional](#traditional)
- [Datasets](#datasets)
- [Tools](#tools)

---


## Coarse Registration

The coarse registration methods (or global registration) aligns two point clouds without an initial guess. We broadly classified these methods into feature matching based, end-to-end, randomized and probabilistic. Most of the learning based methods are focusing on some specific step in the feature matching based algorithms.

### Feature Matching Based

The feature-matching based registration algorithms generally follow a two-stage workflow: determining correspondence and estimate the transformation. The correspondence establishing stage usually follow the four-step pipeline: keypoint detection, feature description, descriptor matching and outlier rejection. The nearest neighbor matching is the de-facto matching strategy, but could be replaced by learnable matching stategies. We also include some papers which adopt the graph algorithms for the matching and outlier rejection problem.

#### Keypoint Detection

- HKS: A Concise and Provably Informative Multi‐Scale Signature Based on Heat Diffusion. CGF'2009 [[paper]](http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Sun09.pdf)
- Harris3D: a robust extension of the harris operator for interest point detection on 3D meshes. VC'2011 [[paper]](http://repositorio.conicyt.cl/themes/Mirage2/tutorial/guia_busquedas_avanzadas.pdf)
- Intrinsic shape signatures: A shape descriptor for 3D object recognition. ICCV'2012 [[paper]](https://www.computer.org/csdl/proceedings/iccvw/2009/4442/00/05457637.pdf)
- Learning a Descriptor-Specific 3D Keypoint Detector. ICCV'2015 [[paper]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Salti_Learning_a_Descriptor-Specific_ICCV_2015_paper.pdf)
- 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration. ECCV'2018 [[paper]](https://arxiv.org/pdf/1807.09413.pdf) [[code]](https://github.com/yewzijian/3DFeatNet)
- USIP: Unsupervised Stable Interest Point Detection from 3D Point Clouds. ICCV'2019 [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_USIP_Unsupervised_Stable_Interest_Point_Detection_From_3D_Point_Clouds_ICCV_2019_paper.pdf) [[code]](https://github.com/lijx10/USIP)
- D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features. CVPR'2020 [[paper]](https://arxiv.org/abs/2003.03164) [[code]](https://github.com/XuyangBai/D3Feat)
- PointCloud Saliency Maps. ICCV'2019 [[paper]](http://arxiv.org/pdf/1812.01687) [[code]](https://github.com/tianzheng4/PointCloud-Saliency-Maps)
- SK-Net: Deep Learning on Point Cloud via End-to-end Discovery of Spatial Keypoints. AAAI'2020 [[paper]](https://arxiv.org/pdf/2003.14014.pdf)
- SKD: Unsupervised Keypoint Detecting for Point Clouds using Embedded Saliency Estimation. arxiv'2019 [[paper]](https://arxiv.org/pdf/1912.04943.pdf)
- Fuzzy Logic and Histogram of Normal Orientation-based 3D Keypoint Detection For Point Clouds. PRL'2020 [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S016786552030180X)
- MaskNet: A Fully-Convolutional Network to Estimate Inlier Points. 3DV'2020 [[paper]](https://arxiv.org/abs/2010.09185) [[code]](https://github.com/vinits5/masknet)
- PREDATOR: Registration of 3D Point Clouds with Low Overlap. arxiv'2020 [[paper]](https://arxiv.org/pdf/2011.13005.pdf) [[code]](https://github.com/ShengyuH/OverlapPredator)


Survey:
- Performance Evaluation of 3D Keypoint Detectors. IJCV'2013 [[paper]](https://doi.org/10.1007/s11263-012-0545-4)

#### Feature Description
- Spin Image: Using spin images for efficient object recognition in cluttered 3D scenes. TPAMI'1999 [[paper]](https://pdfs.semanticscholar.org/30c3/e410f689516983efcd780b9bea02531c387d.pdf?_ga=2.267321353.662069860.1609508014-1451995720.1602238989)
- USC: Unique shape context for 3D data description. 3DOR'2010 [[paper]](http://www.vision.deis.unibo.it/fede/papers/3dor10.pdf)
- 3DShapeContext: Recognizing Objects in Range Data Using Regional Point Descriptors. ECCV'2004 [[paper]](http://www.ri.cmu.edu/pub_files/pub4/frome_andrea_2004_1/frome_andrea_2004_1.pdf)
- SHOT: Unique Signatures of Histograms for Local Surface Description. ECCV'2010 [[paper]](http://www.researchgate.net/profile/Samuele_Salti/publication/262111100_SHOT_Unique_Signatures_of_Histograms_for_Surface_and_Texture_Description/links/541066b00cf2df04e75d5939.pdf)
- FPFH: Fast Point Feature Histograms (FPFH) for 3D registration. ICRA'2009 [[paper]](http://www.cvl.iis.u-tokyo.ac.jp/class2016/2016w/papers/6.3DdataProcessing/Rusu_FPFH_ICRA2009.pdf)
- RoPS: 3D Free Form Object Recognition using Rotational Projection Statistics. WACV'2013 [[paper]](http://www.researchgate.net/profile/Ferdous_Sohel/publication/236645183_3D_free_form_object_recognition_using_rotational_projection_statistics/links/0deec518a1038a2980000000.pdf)
- CGF: Learning Compact Geometric Features. ICCV'2017 [[paper]](http://arxiv.org/pdf/1709.05056)
- 3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions. CVPR'2017 [[paper]](http://arxiv.org/pdf/1603.08182) [[code]](https://github.com/andyzeng/3dmatch-toolbox)
- End-to-end learning of keypoint detector and descriptor for pose invariant 3D matching. CVPR'2018 [[paper]](http://arxiv.org/pdf/1802.07869)
- PPFNet: Global Context Aware Local Features for Robust 3D Point Matching. CVPR'2018 [[paper]](http://arxiv.org/pdf/1802.02669)
- 3DFeat-Net: Weakly Supervised Local 3D Features for Point Cloud Registration. ECCV'2018 [[paper]](https://arxiv.org/pdf/1807.09413.pdf) [[code]](https://github.com/yewzijian/3DFeatNet)
- MVDesc: Learning and Matching Multi-View Descriptors for Registration of Point Clouds. ECCV'2018 [[paper]](https://arxiv.org/pdf/1807.05653.pdf) [[code]](https://github.com/zlthinker/RMBP)
- FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation. CVPR'2018 [[paper]](http://arxiv.org/pdf/1712.07262) [[code]](https://github.com/XuyangBai/FoldingNet)
- PPF-FoldNet: Unsupervised Learning of Rotation Invariant 3D Local Descriptors. ECCV'2018 [[paper]](https://arxiv.org/abs/1808.10322) [[code]](https://github.com/XuyangBai/PPF-FoldNet)
- 3D Local Features for Direct Pairwise Registration. CVPR'2019 [[paper]](https://arxiv.org/abs/1904.04281)
- 3D Point-Capsule Networks. CVPR'2019 [[paper]](https://arxiv.org/abs/1812.10775) [[code]](https://github.com/yongheng1991/3D-point-capsule-networks)
- The Perfect Match: 3D Point Cloud Matching with Smoothed Densities. CVPR'2019 [[paper]](https://arxiv.org/abs/1811.06879) [[code]](https://github.com/zgojcic/3DSmoothNet)
- FCGF: Fully Convolutional Geometric Features. ICCV'2019 [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choy_Fully_Convolutional_Geometric_Features_ICCV_2019_paper.pdf) [[code]](https://github.com/chrischoy/FCGF)
- Learning an Effective Equivariant 3D Descriptor Without Supervision. ICCV'2019 [[paper]](https://arxiv.org/abs/1909.06887)
- D3Feat: Joint Learning of Dense Detection and Description of 3D Local Features. CVPR'2020 [[paper]](https://arxiv.org/abs/2003.03164) [[code]](https://github.com/XuyangBai/D3Feat)
- End-to-End Learning Local Multi-view Descriptors for 3D Point Clouds. CVPR'2020 [[paper]](https://arxiv.org/abs/2003.05855) [[code]](https://github.com/craigleili/3DLocalMultiViewDesc)
- LRF-Net- Learning Local Reference Frames for 3D Local Shape Description and Matching. arxiv'2020 [[paper]](https://arxiv.org/abs/2001.07832)
- DH3D: Deep Hierarchical 3D Descriptors for Robust Large-Scale 6DoF Relocalization. ECCV'2020 [[paper]](https://arxiv.org/pdf/2007.09217.pdf) [[code]](https://github.com/JuanDuGit/DH3D)
- Distinctive 3D local deep descriptors. arxiv'2020 [[paper]](https://arxiv.org/abs/2009.00258) [[code]](https://github.com/fabiopoiesi/dip)
- SpinNet: Learning a General Surface Descriptor for 3D Point Cloud Registration. CVPR'2021 [[paper]](https://arxiv.org/abs/2011.12149) [[code]](https://github.com/QingyongHu/SpinNet)
- PREDATOR: Registration of 3D Point Clouds with Low Overlap. CVPR'2021 [[paper]](https://arxiv.org/pdf/2011.13005.pdf) [[code]](https://github.com/ShengyuH/OverlapPredator)
- Self-supervised Geometric Perception. CVPR'2021 [[paper]](https://arxiv.org/abs/2103.03114) [[code]](https://github.com/theNded/SGP)
- 3D Point Cloud Registration with Multi-Scale Architecture and Self-supervised Fine-tuning. arxiv'2021 [[paper]](https://arxiv.org/abs/2103.14533) [[code]](https://github.com/humanpose1/MS-SVConv)
- Generalisable and Distinctive (GeDi) 3D local deep descriptors for point cloud registration. arxiv'2021 [[paper]](https://arxiv.org/pdf/2105.10382.pdf) [[code]](https://github.com/fabiopoiesi/gedi)
- Neighborhood Normalization for Robust Geometric Feature Learning. CVPR'2021 [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Neighborhood_Normalization_for_Robust_Geometric_Feature_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/lppllppl920/NeighborhoodNormalization-Pytorch)

Survey:
- A Comprehensive Performance Evaluation of 3D Local Feature Descriptors. IJCV'2015 [[paper]](https://link.springer.com/article/10.1007/s11263-015-0824-y)
- Evaluating Local Geometric Feature Representations for 3D Rigid Data Matching. ICIP'2019 [[paper]](https://arxiv.org/abs/1907.00233)

#### Outlier Rejection
- RANSAC: Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography. 1981 [[paper]](http://www.cs.ait.ac.th/~mdailey/cvreadings/Fischler-RANSAC.pdf)
- Locally Optimized RANSAC. 2003 [[paper]](ftp://cmp.felk.cvut.cz/pub/cmp/articles/matas/chum-dagm03.pdf)
- Graph-cut RANSAC. CVPR'2018 [[paper]](https://arxiv.org/abs/1706.00984) [[code]](https://github.com/danini/graph-cut-ransac)
- MAGSAC: Marginalizing Sample Consensus. CVPR'2019 [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Barath_MAGSAC_Marginalizing_Sample_Consensus_CVPR_2019_paper.pdf) [[code]](https://github.com/danini/magsac)
- VFC: A Robust Method for Vector Field Learning with Application To Mismatch Removing. CVPR'2011 [[paper]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.721.5913&rep=rep1&type=pdf)
- In Search of Inliers: 3D Correspondence by Local and Global Voting. CVPR'2014 [[paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Buch_In_Search_of_2014_CVPR_paper.pdf)
- FGR: Fast Global Registration. ECCV'2016 [[paper]](https://vladlen.info/publications/fast-global-registration/) [[code]](https://github.com/intel-isl/FastGlobalRegistration)
- Ranking 3D Feature Correspondences Via Consistency Voting. PRL'2019 [[paper]](https://doi.org/10.1016/j.patrec.2018.11.018)
- An Accurate and Efficient Voting Scheme for a Maximally All-Inlier 3D Correspondence Set. TPAMI'2020 [[paper]](https://ieeexplore.ieee.org/ielx7/34/4359286/08955806.pdf) 
- GORE: Guaranteed Outlier Removal for Point Cloud Registration with Correspondences. TPAMI'2018 [[paper]](https://arxiv.org/abs/1711.10209) [[code]](https://cs.adelaide.edu.au/~aparra/project/gore/)
- A Polynomial-time Solution for Robust Registration with Extreme Outlier Rates. RSS'2019 [[paper]](https://arxiv.org/abs/1903.08588)
- Graduated Non-Convexity for Robust Spatial Perception: From Non-Minimal Solvers to Global Outlier Rejection. ICRA'2020 [[paper]](https://arxiv.org/abs/1909.08605)
- TEASER: Fast and Certifiable Point Cloud Registration. T-RO'2020 [[paper]](https://arxiv.org/abs/2001.07715) [[code]](https://github.com/MIT-SPARK/TEASER-plusplus)
- One Ring to Rule Them All: Certifiably Robust Geometric Perception with Outliers. NeurIPS'2020 [[paper]](https://arxiv.org/abs/2006.06769)
- SDRSAC: Semidefinite-Based Randomized Approach for Robust Point Cloud Registration without Correspondences. CVPR'2019 [[paper]](https://arxiv.org/abs/1904.03483) [[code]](https://github.com/intellhave/SDRSAC)
- Robust Low-Overlap 3D Point Cloud Registration for Outlier Rejection. ICRA'2019 [[paper]](https://arpg.colorado.edu/papers/hmrf_icp.pdf)
- ICOS: Efficient and Highly Robust Rotation Search and Point Cloud Registration with Correspondences. arxiv'2021 [[paper]](https://arxiv.org/pdf/2104.14763.pdf)

Learning based (including 2D outlier rejection methods)
- Learning to Find Good Correspondences. CVPR'2018 [[paper]](https://arxiv.org/abs/1711.05971) [[code]](https://github.com/vcg-uvic/learned-correspondence-release)
- NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences. CVPR'2019 [[paper]](https://arxiv.org/abs/1904.00320) [[code]](https://github.com/sailor-z/NM-Net)
- OANet: Learning Two-View Correspondences and Geometry using Order-Aware Network. ICCV'2019 [[paper]](https://arxiv.org/abs/1908.04964) [[code]](https://github.com/zjhthu/OANet)
- ACNe: Attentive Context Normalization for Robust Permutation-Equivariant Learning. CVPR'2020 [[paper]](https://arxiv.org/abs/1907.02545) [[code]](https://github.com/vcg-uvic/acne)
- SuperGlue: Learning Feature Matching with Graph Neural Networks. CVPR'2020 [[paper]](https://arxiv.org/abs/1911.11763) [[code]](https://github.com/magicleap/SuperGluePretrainedNetwork)
- 3DRegNet: A Deep Neural Network for 3D Point Registration. CVPR'2020 [[paper]](https://arxiv.org/abs/1904.01701) [[code]](https://github.com/goncalo120/3DRegNet)
- Deep Global Registration. CVPR'2020 [[paper]](https://arxiv.org/abs/2004.11540) [[code]](https://github.com/chrischoy/DeepGlobalRegistration)
- 3D Correspondence Grouping with Compatibility Features. arxiv'2020 [[paper]](https://arxiv.org/pdf/2007.10570.pdf)
- PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency CVPR'2021 [[paper]](https://arxiv.org/abs/2103.05465) [[code]](https://github.com/XuyangBai/PointDSC)

Survey
- A Performance Evaluation of Correspondence Grouping Methods for 3D Rigid Data Matching. TPAMI'2019 [[paper]](http://arxiv.org/pdf/1907.02890)


#### Graph Algorithms

- A Graduated Assignment Algorithm for Graph Matching. TPAMI'1996 [[paper]](https://pdfs.semanticscholar.org/9899/003369af99d02f699cbcbf48b79019666158.pdf?_ga=2.31476793.662069860.1609508014-1451995720.1602238989)
- A Spectral Technique for Correspondence Problems Using Pairwise Constraints. ICCV'2005 [[paper]](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/leordeanu-iccv-05.pdf) [[code]](https://sites.google.com/site/graphmatchingmethods/Code_including_Spectral_Matching.zip?attredirects=0)
- Balanced Graph Matching. NIPS'2006 [[paper]](http://papers.nips.cc/paper/2960-balanced-graph-matching) [[code]](https://github.com/afiliot/Balanced-Graph-Matching)
- Feature Correspondence via Graph Matching: Models and Global Optimization. ECCV'2008 [[paper]](https://www.microsoft.com/en-us/research/wp-content/uploads/2008/10/eccv08-MatchingMRF.pdf)
- An Integer Projected Fixed Point Method for Graph Matching and MAP Inference. NIPS'2009 [[paper]](https://www.cs.cmu.edu/~rahuls/pub/nips2009-rahuls.pdf) [[code]](https://sites.google.com/site/graphmatchingmethods/Code_including_IPFP.zip?attredirects=0)
- Optimal Correspondences From Pairwise Constraints. ICCV'2009 [[paper]](https://lup.lub.lu.se/search/ws/files/4311163/1454018.pdf)
- Reweighted Random Walks for Graph Matching. ECCV'2010 [[paper]](https://cv.snu.ac.kr/research/~RRWM/)
- Maximal Cliques Based Rigid Body Motion Segmentation with a RGB-D Camera. ACCV'2012 [[paper]](https://doi.org/10.1007/978-3-642-37444-9_10)
- A Probabilistic Approach to Spectral Graph Matching. TPAMI'2013 [[paper]](http://yosi-keller.narod.ru/publications/pdf/probabilistic-matching_rev3_two_columns.pdf)
- A Practical Maximum Clique Algorithm for Matching with Pairwise Constraints. arxiv'2019 [[paper]](https://arxiv.org/pdf/1902.01534.pdf)
- ROBIN: a Graph-Theoretic Approach to Reject Outliers in Robust Estimation using Invariants. arxiv'2020 [[paper]](https://arxiv.org/abs/2011.03659)
- CLIPPER A Graph-Theoretic Framework for Robust Data Association. arxiv'2020 [[paper]](https://arxiv.org/pdf/2011.10202.pdf)
- PointDSC: Robust Point Cloud Registration using Deep Spatial Consistency CVPR'2021 [[paper]](https://arxiv.org/abs/2103.05465) [[code]](https://github.com/XuyangBai/PointDSC)
- Pairwise Point Cloud Registration Using Graph Matching and Rotation-invariant Features. arxiv'2021 [[paper]](https://arxiv.org/pdf/2105.02151.pdf)


### End-to-End

Some papers perform end-to-end registration by directly predicting a rigid transformation aligning two point clouds without explicitly following the `detection -- description -- outlier filtering` pipepline. While they works well on object-centric datasets, the performance on real-world scene registration is not satisfactory.

- PointNetLK: Robust & Efficient Point Cloud Registration using PointNet. CVPR'2019 [[paper]](https://arxiv.org/abs/1903.05711) [[code]](https://github.com/hmgoforth/PointNetLK)
- Deep Closest Point: Learning Representations for Point Cloud Registration. ICCV'2019 [[paper]](https://arxiv.org/abs/1905.03304) [[code]](https://github.com/WangYueFt/dcp)
- PRNet: Self-Supervised Learning for Partial-to-Partial Registration. NeurIPS'2019 [[paper]](https://arxiv.org/abs/1910.12240) [[code]](https://github.com/WangYueFt/prnet)
- AlignNet-3D: Fast Point Cloud Registration of Partially Observed Objects. 3DV'2019 [[paper]](https://arxiv.org/abs/1910.04668) [[code]](https://github.com/grossjohannes/AlignNet-3D)
- RPM-Net: Robust Point Matching using Learned Features. CVPR'2020 [[paper]](https://arxiv.org/abs/2003.13479) [[code]](https://github.com/yewzijian/RPMNet)
- Feature-metric Registration: A Fast Semi-supervised Approach for Robust Point Cloud Registration without Correspondences. CVPR'2020 [[code]](https://arxiv.org/abs/2005.01014) [[code]](https://github.com/XiaoshuiHuang/fmr)
- Iterative Distance-Aware Similarity Matrix Convolution with Mutual-Supervised Point Elimination for Efficient Point Cloud Registration. ECCV'2020 [[paper]](http://arxiv.org/abs/1910.10328) [[code]](https://github.com/jiahaowork/idam)
- Self-supervised Point Set Local Descriptors for Point Cloud Registration. arxiv'2020 [[paper]](https://arxiv.org/pdf/2003.05199.pdf)
- Learning 3D-3D Correspondences for One-shot Partial-to-partial Registration. arxiv'2020 [[paper]](https://arxiv.org/pdf/2006.04523.pdf)
- Robust Point Cloud Registration Framework Based on Deep Graph Matching. CVPR'2021 [[paper]](https://arxiv.org/abs/2103.04256) [[code]](https://github.com/fukexue/RGM)
- RPSRNet: End-to-End Trainable Rigid Point Set Registration Network using Barnes-Hut 2^D-Tree Representation. CVPR'2021 [[paper]](https://arxiv.org/pdf/2104.05328.pdf)
- Deep Weighted Consensus (DWC) Dense correspondence confidence maps for 3D shape registration. arxiv'2021 [[paper]](https://arxiv.org/pdf/2105.02714.pdf)
- OMNet: Learning Overlapping Mask for Partial-to-Partial Point Cloud Registration. arxiv'2021 [[paper]](https://arxiv.org/pdf/2103.00937.pdf)
- FINet: Dual Branches Feature Interaction for Partial-to-Partial Point Cloud Registration. arxiv'2021 [[paper]](https://arxiv.org/pdf/2106.03479.pdf)
- PointNetLK Revisited. CVPR'2021 [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_PointNetLK_Revisited_CVPR_2021_paper.pdf)

### Randomized 

- RANSAC: Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography. 1981 [[paper]](http://www.cs.ait.ac.th/~mdailey/cvreadings/Fischler-RANSAC.pdf)
- 4PCS: 4-points Congruent Sets for Robust Pairwise Surface Registration. TOG'2008 [[paper]](http://www.cs.bgu.ac.il/~aiger/a85-aiger.pdf)
- Model Globally, Match Locally: Efficient and Robust 3D Object Recognition. CVPR'2010 [[paper]](http://campar.cs.tum.edu/pub/drost2010CVPR/drost2010CVPR.pdf) [[code]](https://github.com/adrelino/ppf-reconstruction)
- Super 4PCS: Fast Global Pointcloud Registration via Smart Indexing. CGF'2014 [[paper]](https://hal.archives-ouvertes.fr/hal-01538738/file/super4pcs.pdf) [[code]](https://github.com/nmellado/Super4PCS)

### Probabilistic

- Point Set Registration: Coherent Point Drift. TPAMI'2010 [[paper]](https://arxiv.org/pdf/0905.2635.pdf) [[code]](https://github.com/neka-nat/probreg)
- Robust Point Set Registration Using Gaussian Mixture Models. TPAMI'2011 [[paper]](https://github.com/bing-jian/gmmreg/blob/master/gmmreg_PAMI_preprint.pdf) [[code]](https://github.com/bing-jian/gmmreg)
- A Generative Model for the Joint Registration of Multiple Point Sets. ECCV'2014 [[paper]](http://hal.inria.fr/docs/01/05/45/69/PDF/main_cr.pdf)
- Aligning the Dissimilar: A Probabilistic Method for Feature-based Point Set Registration. ICPR'2016 [[paper]](http://liu.diva-portal.org/smash/get/diva2:1104306/FULLTEXT01)
- A Probabilistic Framework for Color-Based Point Set Registration. CVPR'2016 [[paper]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Danelljan_A_Probabilistic_Framework_CVPR_2016_paper.pdf)
- Density Adaptive Point Set Registration. CVPR'2018 [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Lawin_Density_Adaptive_Point_CVPR_2018_paper.pdf) [[code]](https://github.com/felja633/DARE)
- HGMR: Hierarchical Gaussian Mixtures for Adaptive 3D Registration. ECCV'2018 [[paper]](http://jankautz.com/publications/hGMM_ECCV18.pdf)
- Robust Feature-Based Point Registration Using Directional Mixture Model. arxiv'2019 [[paper]](https://arxiv.org/pdf/1912.05016.pdf)
- FilterReg: Robust and Efficient Probabilistic Point-Set Registration using Gaussian Filter and Twist Parameterization. CVPR'2019 [[paper]](https://arxiv.org/abs/1811.10136) [[code]](https://bitbucket.org/gaowei19951004/poser/src/master/)
- PointGMM: a Neural GMM Network for Point Clouds. CVPR'2020 [[paper]](https://arxiv.org/abs/2003.13326) [[code]](https://github.com/amirhertz/pointgmm)
- DeepGMR: Learning Latent Gaussian Mixture Models for Registration. ECCV'2020 [[paper]](https://arxiv.org/abs/2008.09088) [[code]](https://github.com/wentaoyuan/deepgmr)
- Registration Loss Learning for Deep Probabilistic Point Set Registration. 3DV'2020 [[paper]](https://arxiv.org/abs/2011.02229) [[code]](https://github.com/felja633/RLLReg)
- A Termination Criterion for Probabilistic PointClouds Registration. arxiv'2020 [[paper]](https://arxiv.org/pdf/2010.04979.pdf)
- LSG-CPD: Coherent Point Drift with Local Surface Geometry for Point Cloud Registration. arxiv'2021 [[paper]](https://arxiv.org/pdf/2103.15039.pdf)

### Others
- Gravitational Approach for Point Set Registration. CVPR'2016 [[paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Golyanik_Gravitational_Approach_for_CVPR_2016_paper.pdf)
- Accelerated Gravitational Point Set Alignment with Altered Physical Laws. ICCV'2019 [[paper]](https://people.mpi-inf.mpg.de/~golyanik/04_DRAFTS/Golyanik_etal_ICCV_2019.pdf)
- Fast Gravitational Approach for Rigid Point Set Registration with Ordinary Differential Equations. arxiv'2020 [[paper]](https://arxiv.org/abs/2009.14005)
- Minimal Solvers for Mini-Loop Closures in 3D Multi-Scan Alignment. CVPR'2019 [[paper]](http://arxiv.org/pdf/1904.03941)
- Minimal Solvers for 3D Scan Alignment With Pairs of Intersecting Lines. CVPR'2020 [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mateus_Minimal_Solvers_for_3D_Scan_Alignment_With_Pairs_of_Intersecting_CVPR_2020_paper.pdf)
- Learning multiview 3D point cloud registration. CVPR'2020 [[paper]](https://arxiv.org/abs/2001.05119) [[code]](https://github.com/zgojcic/3D_multiview_reg)
- A Dynamical Perspective on Point Cloud Registration. arxiv'2020 [[paper]](https://arxiv.org/abs/2005.03190)
- Plane Pair Matching for Efficient 3D View Registration. arxiv'2020 [[paper]](https://arxiv.org/pdf/2001.07058.pdf)

## Fine Registration

The fine registration methods (or local registration) produce highly precise registration results, given the initial pose between two point clouds. 

### Traditional 
- Point2Point ICP: A Method for Registration of 3-D Shapes. TPAMI'1992 [[paper]](https://ieeexplore.ieee.org/document/121791)
- Point2Plane Object Modelling by Registration of Multiple Range Images. TPAMI'1992 [[paper]](http://www.cs.hunter.cuny.edu/~ioannis/chen_medioni_point_plane_1991.pdf)
- RPM: New Algorithms for 2D and 3D Point Matching: Pose Estimation and Correspondence. [[paper]](http://cmp.felk.cvut.cz/~amavemig/softassign.pdf)
- Matching of 3-D Curves using Semi-differential Invariants. ICCV'1995 [[paper]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=466913)
- The Trimmed Iterative Closest Point Algorithm. 2002 [[paper]](http://www.inf.u-szeged.hu/ssip/2002/download/Chetverikov.pdf)
- Comparing ICP Variants on Real-World Data Sets. 2013 [[paper]](https://hal.archives-ouvertes.fr/hal-01143458/file/2013_Pomerleau_AutonomousRobots_Comparing.pdf)
- Generalized-ICP. RSS'2009 [[paper]](https://doi.org/10.15607%2Frss.2009.v.021) [[code]](https://github.com/avsegal/gicp)
- Go-ICP: Solving 3D Registration Efficiently and Globally Optimally. ICCV'2013 [[paper]](http://jlyang.org/iccv13_go-icp.pdf) [[code]](https://github.com/yangjiaolong/Go-ICP)
- AA-ICP: Iterative Closest point with Anderson Acceleration. ICRA'2018 [[paper]](https://arxiv.org/abs/1709.05479#:~:text=Iterative%20Closest%20Point%20(ICP)%20is,performing%20scan%2Dmatching%20and%20registration.&text=This%20method%20is%20based%20on,fixed%20point%20of%20contractive%20mapping.)
- Point Clouds Registration with Probabilistic Data Association. IROS'2016 [[paper]](https://github.com/ethz-asl/ipda/wiki/0383.pdf) [[code]](https://github.com/ethz-asl/robust_point_cloud_registration)
- GH-ICP：Iterative Closest Point Algorithm with Global Optimal Matching and Hybrid Metric. 3DV'2018 [[paper]](https://ieeexplore.ieee.org/abstract/document/8490968) [[code]](https://github.com/YuePanEdward/GH-ICP)
- NDT: The Normal Distributions Transform: A New Approach To Laser Scan Matching. IROS'2003 [[paper]](http://hdl.handle.net/10068/262019)
- Best Buddies Registration for Point Clouds. ACCV'2020 [[paper]](https://openaccess.thecvf.com/content/ACCV2020/papers/Drory_Best_Buddies_Registration_for_Point_Clouds_ACCV_2020_paper.pdf)
- Provably Approximated ICP. arxiv'2021 [[paper]](https://arxiv.org/pdf/2101.03588.pdf)

### Learning-based
- DeepICP: An End-to-End Deep Neural Network for 3D Point Cloud Registration. ICCV'2019 [[paper]](https://arxiv.org/abs/1905.04153)


## Datasets
- [Standford 3DScanning](http://graphics.stanford.edu/data/3Dscanrep/)
- [3DMatch](http://3dmatch.cs.princeton.edu/)
- [ETH (Challenging data sets for point cloud registration algorithms)](https://projects.asl.ethz.ch/datasets/doku.php?id=laserregistration:laserregistration)
- [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [ModelNet](https://modelnet.cs.princeton.edu/)
- [A Benchmark for Point Clouds Registration Algorithms](https://github.com/iralabdisco/point_clouds_registration_benchmark)
- [WHU-TLS Benchmark](http://3s.whu.edu.cn/ybs/en/benchmark.htm)


## Tools
- [Open3D: A Modern Libray for 3D Data Processing](http://www.open3d.org/docs/release/index.html)
- [PCL: Point Cloud Library](https://pointclouds.org/)
- [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric)
- [PyTorch Points 3D](https://github.com/nicolas-chaulet/torch-points3d)
- [probreg](https://github.com/neka-nat/probreg)
