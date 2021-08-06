package it.units.malelab.jgea.core.order;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.datasets.iterator.DoublesDataSetIterator;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.InvalidScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class AuroraMap<T> implements PartiallyOrderedCollection<T> {

    protected final HashMap<List<Double>, T> archive;
    protected final Boolean maximize;
    protected final Function<T, double[]> descriptor;
    protected final Function<T, Double> getFitness;
    protected final PartialComparator<? super T> comparator;
    public ArrayList<T> lastAdded = new ArrayList<>();
    protected int batch_size = 128;
    protected final int size;
    public int counter = 0;
    public int counter2 = 0;
    protected final int k;
    protected final int nc_target;
    protected final Function<T, double[][]> getData;
    protected double minD;
    protected int neighbourSize;
    protected MultiLayerNetwork vae;

    public AuroraMap(int bd_size, int neighbourSize, int k, int nc_target, int batch_size, Boolean maximize,
                     PartialComparator<? super T> comparator, Function<T, Double> getFitness, Function<T, double[][]> getData,
                     MultiLayerConfiguration conf) {
        archive = new HashMap<>();
        this.maximize = maximize;
        this.descriptor = ind -> {
            double[][][][] md = new double[1][1][][];
            md[0][0] = getData.apply(ind);
            INDArray input = Nd4j.create(md);

            /*NativeImageLoader nil = new NativeImageLoader(md.length, md[0].length, 1);
            try {
                input = nil.asMatrix(md);
            } catch (IOException e) {
                e.printStackTrace();
            }*/
            vae.setInputMiniBatchSize(1);
            vae.setInput(input);
            double[] d = vae.activateSelectedLayers(0,4,input).toDoubleVector();

            return d;
        };

        this.getFitness = getFitness;
        this.comparator = comparator;

        this.size = bd_size;

        this.k = k;
        this.neighbourSize = neighbourSize;
        this.batch_size = batch_size;
        this.nc_target = nc_target;
        this.getData = getData;
        this.minD = 0;
        this.vae = new MultiLayerNetwork(conf);
        this.vae.init();
        vae.setListeners(new ScoreIterationListener(1));


    }

    public void trainEncoder(ArrayList<Pair<INDArray, INDArray>> data) {
        long tt =System.currentTimeMillis();
        System.out.println("start training "+tt);
        vae.setInputMiniBatchSize(32);
        INDArrayDataSetIterator dd = new INDArrayDataSetIterator(data, batch_size);
        DataSetIteratorSplitter ds = new DataSetIteratorSplitter(dd, data.size() % batch_size, 0.8);

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(10000))
                .iterationTerminationConditions(new InvalidScoreIterationTerminationCondition())
                .scoreCalculator(new DataSetLossCalculator(ds.getTestIterator(), true))
                .evaluateEveryNEpochs(1)
                .build();
        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, vae, ds.getTrainIterator());
        EarlyStoppingResult t = trainer.fit();
        System.out.println("end training "+(System.currentTimeMillis()-tt));
        System.out.println(t.getTerminationDetails());
    }

    public void updateArchive(ArrayList<T> pop) {


        for (int i = 0; i < pop.size(); i++) {
            add(pop.get(i));
        }
        lastAdded.clear();

    }

    public RealMatrix getDescriptors(ArrayList<T> pop) {
        RealMatrix desc = MatrixUtils.createRealMatrix(pop.size(), size);

        int r = 0;
        for (T individual : pop) {

            desc.setRow(r,descriptor.apply(individual));
        }

        return desc;
    }

    public void updateDescriptors() {
        ArrayList<Pair<INDArray,INDArray>> data = new ArrayList<>(); //rows individual, columns are sensory data
        ArrayList<T> pop = new ArrayList<>(archive.values());
        archive.clear();

        int r = 0;
        for (T individual : pop) {
            INDArray ind_data =  Nd4j.create(getData.apply(individual));
            System.out.println(ind_data.columns()+"   dd  "+ind_data.rows());
            data.add(new Pair<>(ind_data, ind_data));
            r += 1;
        }

        trainEncoder(data);
        data.clear();

        RealMatrix newDescriptor = getDescriptors(pop);
        //System.out.println(newDescriptor.getRowDimension()+"   "+newDescriptor.getColumnDimension());
        this.minD = newMind(newDescriptor);
        System.out.println(this.minD);
        updateArchive(pop);


    }

    public void initialiseMinDistance(Collection<T> individuals) {
        double[][] beavs = new double[individuals.size()][];
        int c = 0;
        for (T ind: individuals) {
            beavs[c] = descriptor.apply(ind);
            c++;

        }

        for (int i = 0; i < beavs.length; i++) {
            double avg = Arrays.stream(beavs[i]).average().getAsDouble();
            beavs[i] = Arrays.stream(beavs[i]).map(d -> d - avg).toArray();
        }

        RealMatrix normed = MatrixUtils.createRealMatrix(beavs);
        EigenDecomposition ed = new EigenDecomposition(normed.transpose().multiply(normed));
        RealMatrix ev = ed.getV();
        normed = (ev.transpose().multiply(normed.transpose())).transpose();

        RealMatrix finalDescs = normed;

        double volume = IntStream.range(0, normed.getColumnDimension()).
                mapToDouble(i ->
                        Arrays.stream(finalDescs.getColumn(i)).max().getAsDouble() -
                                Arrays.stream(finalDescs.getColumn(i)).min().getAsDouble())
                .reduce(1, (a, b) -> a * b);
        this.minD = 0.5 * nroot(volume / nc_target, size);
        System.out.println("initial min distance "+this.minD);
    }

    private double nroot(double val, double n) {
        return Math.exp(Math.log(val) / n);
    }

    private double newMind(RealMatrix descriptors) {
        double maxDistance = distance(descriptors);
        System.out.println("max distance "+maxDistance);
        double tmp = this.k * this.nc_target;
        return maxDistance / nroot(k, size);
    }

    private double mindCorrection() {
        return 0;
    }

    private double distance(RealMatrix data) {
        //System.out.println("data "+data.getColumnDimension()+" "+data.getRowDimension());
        RealMatrix xx = MatrixUtils.createColumnRealMatrix(IntStream.range(0, data.getRowDimension()).mapToDouble(i -> Arrays.stream(data.getRow(i)).map(d -> d * d).sum()).toArray());
        RealMatrix xy = (data.scalarMultiply(2)).multiply(data.transpose());
        RealMatrix dist = xx.multiply((MatrixUtils.createRealMatrix(1, xx.getRowDimension()).scalarAdd(1)));
        dist = dist.add((MatrixUtils.createRealMatrix(xx.getRowDimension(), 1).scalarAdd(1)).multiply(xx.transpose()));
        dist = dist.subtract(xy);
        return maxCoeff(dist);

    }

    private double maxCoeff(RealMatrix m) {
        double max = -Double.MAX_VALUE;
        for (int r = 0; r < m.getRowDimension(); r++) {
            for (int c = 0; c < m.getColumnDimension(); c++) {
                double e = m.getEntry(r, c);
                if (max < e) {
                    max = e;
                }
            }
        }
        return max;
    }

    private double pointDistance(T i, T i1) {
        RealMatrix mat = MatrixUtils.createRealMatrix(2, size);

        mat.setRow(0, descriptor.apply(i));
        mat.setRow(1, descriptor.apply(i1));
        return distance(mat);
    }

    private double nearestDistance(T ind) {
        return pointDistance(ind, knn(ind, 1).get(0)); //nearest excluding itself

    }

    private double novelty(T ind, List<T> nn) {//avg distance
        double novelty = 0;
        for (int i = 0; i < nn.size(); i++) {
            novelty += pointDistance(nn.get(i), ind);
        }
        return novelty / nn.size();
    }

    private List<T> knn(T ind, int k) {
        TreeMap<Double, T> nn = new TreeMap<>();
        for (T ind1 : archive.values()) {
            if (!ind.equals(ind1)) {
                double dist = pointDistance(ind, ind1);
                nn.put(dist, ind1);
                if (nn.size() > k) {
                    nn.pollLastEntry();
                }
            }
        }
        return new ArrayList<>(nn.values());
    }

    public Collection<T> values() {
        Collection<T> pops = archive.values();
        return pops;
    }

    @Override
    public Collection<T> all() {
        return Collections.unmodifiableCollection(archive.values());
    }

    @Override
    public Collection<T> firsts() {
        return all();
    }

    @Override
    public Collection<T> lasts() {
        return all();
    }

    @Override
    public boolean remove(T t) {
        double[] indexes = this.descriptor.apply(t);
        return archive.remove(Arrays.stream(indexes).boxed().collect(Collectors.toList())) != null;

    }

    public void addAll(Collection<T> indvs) {
        for (T ind : indvs) {
            add(ind);
        }
    }

    public void saveEncoder(String filename)  {
        try {
            vae.save(new File(filename));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void add(T ind) {
        if (archive.size() == 0 || nearestDistance(ind) > this.minD) {
            archive.put(Arrays.stream(descriptor.apply(ind)).boxed().collect(Collectors.toList()), ind);
            lastAdded.add(ind);
        } else if (archive.size() == 1) {
        } else {
            List<T> nn = knn(ind, 2);
            if (pointDistance(ind, nn.get(1)) > 0.9 * this.minD) {

                T ind1 = nn.get(0);
                double[] score_ind = new double[2];
                double[] score_ind1 = new double[2];
                score_ind[0] = getFitness.apply(ind); // fitness
                score_ind1[0] = getFitness.apply(ind1); // fitness

                nn = knn(ind, this.neighbourSize);
                List<T> nn1 = knn(ind1, this.neighbourSize);

                score_ind[1] = novelty(ind, nn.subList(1, nn.size())); //calc novelty excluding the other candidate
                score_ind1[1] = novelty(ind1, nn1.subList(1, nn1.size()));

                if ((score_ind[0] >= (1 - Math.signum(score_ind1[0]) * 0.1) * score_ind1[0] &&
                        score_ind[1] >= (1 - Math.signum(score_ind1[1]) * 0.1) * score_ind1[1]) &&
                        ((score_ind[0] - score_ind1[0]) * Math.abs(score_ind1[1]) > -(score_ind[1] - score_ind1[1]) * Math.abs(score_ind1[0]))) {

                    archive.remove(Arrays.stream(descriptor.apply(ind1)).boxed().collect(Collectors.toList()));
                    archive.put(Arrays.stream(descriptor.apply(ind)).boxed().collect(Collectors.toList()), ind);
                    lastAdded.add(ind);
                }
            }

        }
    }
}
