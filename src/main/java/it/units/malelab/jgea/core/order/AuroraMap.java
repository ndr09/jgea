package it.units.malelab.jgea.core.order;

import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.util.Pair;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.correlation.Covariance;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.BiFunction;
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
    public ArrayList<T> lastRemoved = new ArrayList<>();
    protected int batch_size = 128;
    protected final int size;
    public int counter = 0;
    public int counter1 =0;
    public int counter2 = 0;
    protected final int k;
    protected final int nc_target;
    protected final Function<T, double[]> getData;
    protected double minD;
    protected int neighbourSize;
    protected RealMatrix eigenvectors;
    protected RealMatrix mean;
    protected  BiFunction<T, double[], double[]> setDesc;
    protected int fs;

    public AuroraMap(int bd_size, int neighbourSize, int k, int nc_target, int batch_size, Boolean maximize, int fs,
                     PartialComparator<? super T> comparator, Function<T, Double> getFitness, Function<T, double[]> getData, BiFunction<T, double[], double[]> setDesc
    ) {
        archive = new HashMap<>();
        this.maximize = maximize;
        this.fs =fs;
        this.setDesc =setDesc;
        this.descriptor = ind -> {
            long tt = System.currentTimeMillis();
            //System.out.println("start desc");
            double[] pointsArray = getData.apply(ind);
            RealMatrix realMatrix = MatrixUtils.createRowRealMatrix(pointsArray);
            if (this.mean == null) {
                calculateMean(realMatrix);
            }
            meanCenterData(realMatrix);
            //create real matrix
            double[][] desc = eigenvectors.transpose().multiply(realMatrix.transpose()).transpose().getData();

            return desc[0];
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


    }

    private void calculateMean(RealMatrix data) {
        if (mean == null) {
            //System.out.println("mean dimension "+data.getColumnDimension());
            mean = MatrixUtils.createRealMatrix(1, data.getColumnDimension());
        }
        for (int i = 0; i < data.getColumnDimension(); i++) {
            mean.setEntry(0, i, Arrays.stream(data.getColumn(i)).average().getAsDouble());
        }

    }

    public void trainEncoder(RealMatrix data) {
        calculateMean(data);
        meanCenterData(data);
        EigenDecomposition ed = new EigenDecomposition(data.transpose().multiply(data));
        if (eigenvectors==null){
            eigenvectors = MatrixUtils.createRealMatrix(fs,size);
        }
        for (int i = 0; i < size; i++) {
            eigenvectors.setColumnVector(i, ed.getEigenvector(i));
        }
    }

    public void updateArchive(ArrayList<T> pop) {

        addAll(pop);
        //lastAdded.clear();

    }

    public RealMatrix getDescriptors(ArrayList<T> pop) {
        RealMatrix desc = MatrixUtils.createRealMatrix(pop.size(), size);

        int r = 0;
        for (T individual : pop) {

            desc.setRow(r, descriptor.apply(individual));
        }

        return desc;
    }

    public RealMatrix getDatas(Collection<T> pop) {
        RealMatrix desc = MatrixUtils.createRealMatrix(pop.size(), fs);

        int r = 0;
        for (T individual : pop) {

            desc.setRow(r, getData.apply(individual));
        }

        return desc;
    }


    public void updateDescriptors() {
        ArrayList<T> pop = new ArrayList<>(archive.values());
        archive.clear();
        RealMatrix newDescriptor = getDatas(pop);
        trainEncoder(newDescriptor);
        this.minD = newMind(getDescriptors(pop));
        updateArchive(pop);
    }

    public void initialiseMinDistance(Collection<T> individuals) {
        double[][] beavs = new double[individuals.size()][];
        RealMatrix mat = getDatas(individuals);
        trainEncoder(mat);

        int c = 0;
        for (T ind : individuals) {
            beavs[c] = descriptor.apply(ind);
            c++;

        }

        for (int i = 0; i < beavs.length; i++) {
            double avg = Arrays.stream(beavs[i]).average().getAsDouble();
            beavs[i] = Arrays.stream(beavs[i]).map(d -> d - avg).toArray();
        }

        RealMatrix normed = MatrixUtils.createRealMatrix(beavs);
        //EigenDecomposition ed = new EigenDecomposition(normed.transpose().multiply(normed));
        //RealMatrix ev = ed.getV();
        //normed = (ev.transpose().multiply(normed.transpose())).transpose();

        RealMatrix finalDescs = normed;

        double volume = IntStream.range(0, normed.getColumnDimension()).
                mapToDouble(i ->
                        Arrays.stream(finalDescs.getColumn(i)).max().getAsDouble() -
                                Arrays.stream(finalDescs.getColumn(i)).min().getAsDouble())
                .reduce(1, (a, b) -> a * b);
        this.minD = 0.5 * nroot(volume / nc_target, size);
        System.out.println("initial min distance " + this.minD);
    }

    private double nroot(double val, double n) {
        return Math.exp(Math.log(val) / n);
    }

    private double newMind(RealMatrix descriptors) {
        double maxDistance = distance(descriptors);

        double tmp = this.k * this.nc_target;
        double f = maxDistance / nroot(tmp, size);
        System.out.println("new distance " + f+"  "+maxDistance+"  "+tmp+"  "+ nroot(tmp, size));
        return f;
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

        double[] a = descriptor.apply(i);
        double[] b = descriptor.apply(i1);
        double[] sum = new double[a.length];
        for(int index=0; index< a.length; index++){
            sum[index] = Math.pow(a[index]-b[index], a.length);
        }
        return nroot(Arrays.stream(sum).sum(), a.length);
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
        int c = 0;
        for (T ind : indvs) {
            add(ind);
            c++;
        }
        System.out.println("add c "+c+"   not add "+this.counter+"   new add "+this.counter1+"  updated"+this.counter2);
    }

    private void meanCenterData(RealMatrix data) {
        for (int i = 0; i < data.getColumnDimension(); i++) {
            for (int j = 0; j < data.getRowDimension(); j++) {
                data.setEntry(j, i, data.getEntry(j, i) - this.mean.getEntry(0, i));
            }
        }
    }
    /*public void saveEncoder(String filename)  {
        try {
            vae.save(new File(filename));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }*/

    @Override
    public void add(T ind) {
        if (archive.size() == 0 || nearestDistance(ind) > this.minD) {
            double[] desc =descriptor.apply(ind);
            archive.put(Arrays.stream(desc).boxed().collect(Collectors.toList()), ind);
            setDesc.apply(ind,desc);
            lastAdded.add(ind);
            this.counter1 +=1;
        } else if (archive.size() == 1) {
            this.counter += 1;
        } else {
            List<T> nn = knn(ind, 2);
            if (pointDistance(ind, nn.get(1)) > this.minD) {

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
                    double[] desc =descriptor.apply(ind);
                    archive.put(Arrays.stream(desc).boxed().collect(Collectors.toList()), ind);
                    setDesc.apply(ind,desc);
                    lastAdded.add(ind);

                    setDesc.apply(ind1, descriptor.apply(ind1));
                    lastRemoved.add(ind1);

                    this.counter2 += 1;
                }else {
                    this.counter += 1;
                }
            }else{
                this.counter +=1;
            }

        }
    }
}
